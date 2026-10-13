"""python -m perception_server  (from the directory containing this package) starts the server on PERCEPTION_PORT (8100)."""

import os

import uvicorn

if __name__ == '__main__':
    uvicorn.run('perception_server.app:app', host='0.0.0.0', port=int(os.environ.get('PERCEPTION_PORT', '8100')))
