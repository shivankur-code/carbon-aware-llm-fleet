"""Minimal mock of vLLM's OpenAI-compatible streaming /v1/completions, for testing the C2
client without a GPU. Latency grows with concurrency so SLO attainment falls with load."""
import asyncio
import json
import sys

from aiohttp import web

ACTIVE = 0


async def models(_):
    return web.json_response({"data": [{"id": "mock-model"}]})


async def completions(req):
    global ACTIVE
    body = await req.json()
    n_in, n_out = len(body["prompt"]), int(body["max_tokens"])
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(req)
    ACTIVE += 1
    try:
        await asyncio.sleep(0.02 + n_in / 200_000 * (1 + ACTIVE / 8))           # prefill
        for _ in range(n_out):
            await asyncio.sleep(0.002 * (1 + ACTIVE / 16))                      # decode step
            await resp.write(f"data: {json.dumps({'choices': [{'text': 'x'}]})}\n\n".encode())
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': n_out}})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
    finally:
        ACTIVE -= 1
    return resp


app = web.Application()
app.add_routes([web.get("/v1/models", models), web.post("/v1/completions", completions)])
web.run_app(app, port=int(sys.argv[1]) if len(sys.argv) > 1 else 8765, print=None)
