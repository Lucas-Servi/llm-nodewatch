# Remote Client Mode

Use the CLI from any machine to query a remote nodewatch server. The client auto-loads credentials from a `.env` file in the project root.

## Setup

```bash
pip install -e ".[client]"    # installs requests + python-dotenv
cp .env.example .env          # copy template and fill in your values
```

## `.env` configuration

```bash
# Required: remote API URL
NODEWATCH_URL=http://your-server:8000/nodewatch/

# Authentication (choose one):

# Option 1: Pre-obtained token
NODEWATCH_TOKEN=your_access_token

# Option 2: Auto-login via a custom auth module
NODEWATCH_LOGIN_MODULE=your_package.auth
NODEWATCH_LOGIN_FUNCTION=get_token
```

The auto-login option calls `your_package.auth.get_token()` which should return a token string. This lets you implement any authentication scheme without modifying nodewatch code.

## Usage

Once `.env` is configured, all CLI commands work transparently against the remote server:

```bash
nodewatch list-runs --last 5
nodewatch report --graph v2 --last 10
nodewatch inspect <run_id>
```

The `--db` flag is ignored in remote mode. The access token is appended as `?access_token=<token>` to all API requests.

## Authentication resolution order

1. `NODEWATCH_TOKEN` — used directly if set
2. `NODEWATCH_LOGIN_MODULE` + `NODEWATCH_LOGIN_FUNCTION` — dynamically imports and calls the function
3. No authentication — requests are sent without a token

