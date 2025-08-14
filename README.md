# Chat-Server
Built on FastAPI

## Usage

**Note**: the `main` branch is intended as a template, not meant to be built or ran.
Refer to implementation branches as examples.

To begin, populate envvars:

```bash
cp .env.example .env
# inspect .env and populate values
```

### Development Server

Set up virtualenv, and install dependencies. Python3.11+ is required on your host.

```bash
python3 -m venv . venv
. venv/bin/activate
pip install -e .
```

To start a server:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8765
```

### Production

Running with Docker is recommended. 
`Dockerfile` is provided to help with image building.
`docker-compose.yml.example` is provided *as an example* to help with orchestrating the server,
connecting it to an instance of Open WebUI.
