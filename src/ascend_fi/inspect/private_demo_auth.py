"""Protect every private demo operation except read-only health/metrics."""
import hmac
import os


class PrivateDemoAuth:
    def __init__(self, app):
        self.app = app
        self.expected = ('Bearer ' + os.environ['VLLM_API_KEY']).encode()
        if len(self.expected) < 39:
            raise RuntimeError('PRIVATE_DEMO_API_KEY_REQUIRED')

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and scope['path'] not in ('/health', '/metrics'):
            supplied = dict(scope.get('headers', [])).get(b'authorization', b'')
            if not hmac.compare_digest(supplied, self.expected):
                await send({'type': 'http.response.start', 'status': 401,
                            'headers': [(b'content-type', b'application/json')]})
                await send({'type': 'http.response.body', 'body': b'{"error":"unauthorized"}'})
                return
        await self.app(scope, receive, send)
