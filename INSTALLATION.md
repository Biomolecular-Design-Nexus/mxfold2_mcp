# MXfold2 MCP Server Installation Guide

## Overview

This guide provides step-by-step instructions for installing and configuring the MXfold2 MCP server with Claude Code and other MCP-compatible clients.

## Prerequisites

- Python 3.10+
- Claude Code CLI or MCP-compatible client
- Git (for cloning dependencies)

## Quick Start

### 1. Verify Server Installation

```bash
# Navigate to MCP directory
cd /path/to/mxfold2_mcp

# Test server starts
./env/bin/python src/server.py --help
```

You should see the FastMCP startup banner with "Server name: mxfold2".

### 2. Register with Claude Code

```bash
# Get absolute paths
MCP_DIR=$(pwd)
PYTHON_PATH="$MCP_DIR/env/bin/python3.10"
SERVER_PATH="$MCP_DIR/src/server.py"

# Register the server
claude mcp add mxfold2 -- "$PYTHON_PATH" "$SERVER_PATH"
```

### 3. Verify Installation

```bash
# Check server is registered
claude mcp list

# Should show:
# mxfold2: /path/to/env/bin/python3.10 /path/to/src/server.py
```

### 4. Test the Installation

Start Claude Code and test:

```
User: "What tools are available from the mxfold2 server?"
```

Expected response: List of 11 tools including job management and RNA analysis tools.

## Detailed Installation

### For Claude Code

#### Method 1: Automatic Registration

```bash
cd /path/to/mxfold2_mcp
chmod +x install_claude_code.sh  # If you have the script
./install_claude_code.sh
```

#### Method 2: Manual Registration

```bash
# Ensure you're in the MCP directory
cd /path/to/mxfold2_mcp

# Register using absolute paths
claude mcp add mxfold2 -- "$(pwd)/env/bin/python3.10" "$(pwd)/src/server.py"

# Verify
claude mcp list | grep mxfold2
```

#### Troubleshooting Claude Code Installation

**Issue: Command not found**
```bash
# Check Claude Code is installed
which claude
# If not found, install Claude Code CLI
```

**Issue: Server registration fails**
```bash
# Check paths are correct
ls -la "$(pwd)/env/bin/python3.10"
ls -la "$(pwd)/src/server.py"

# Test server manually
./env/bin/python src/server.py --help
```

**Issue: Tools not appearing**
```bash
# Remove and re-add server
claude mcp remove mxfold2
claude mcp add mxfold2 -- "$(pwd)/env/bin/python3.10" "$(pwd)/src/server.py"

# Check Claude Code logs (if available)
```

### For Gemini CLI

#### 1. Locate Configuration File

```bash
# Create config directory if it doesn't exist
mkdir -p ~/.gemini
```

#### 2. Update Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "mxfold2": {
      "command": "/absolute/path/to/mxfold2_mcp/env/bin/python3.10",
      "args": ["/absolute/path/to/mxfold2_mcp/src/server.py"],
      "env": {
        "PYTHONPATH": "/absolute/path/to/mxfold2_mcp"
      }
    }
  }
}
```

**Replace `/absolute/path/to/mxfold2_mcp` with your actual path!**

#### 3. Test Gemini Installation

```bash
# Start Gemini CLI
gemini

# Test in Gemini
> What tools do you have access to from mxfold2?
```

### For Other MCP Clients

The server follows the standard MCP protocol. For any MCP-compatible client:

1. **Command**: `/path/to/mxfold2_mcp/env/bin/python3.10`
2. **Arguments**: `["/path/to/mxfold2_mcp/src/server.py"]`
3. **Working Directory**: `/path/to/mxfold2_mcp`
4. **Environment**: Set `PYTHONPATH` to include the MCP directory

## Configuration Options

### Environment Variables

```bash
# Optional: Set custom job directory
export MCP_JOBS_DIR="/path/to/custom/jobs"

# Optional: Set custom log level
export MCP_LOG_LEVEL="DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

### Server Configuration

Edit `src/server.py` to modify:

- **Server name**: Change `mcp = FastMCP("mxfold2")`
- **Job timeout**: Modify timeout values in job manager
- **Default parameters**: Update DEFAULT_CONFIG in scripts

## Testing Your Installation

### Basic Functionality Test

```bash
cd /path/to/mxfold2_mcp
./env/bin/python tests/run_integration_tests.py
```

Expected output: All tests should pass with 100% success rate.

### Deployment Readiness Test

```bash
./env/bin/python tests/test_deployment_readiness.py
```

Expected output: "Ready for Deployment: ✅ YES" with 100% readiness score.

### Manual Testing

Use these prompts in your MCP client:

#### 1. Tool Discovery
```
"What tools are available from mxfold2? List them with descriptions."
```

#### 2. Sync Tool Test
```
"Use predict_rna_structure on examples/data/sample_rna.fa with Turner model"
```

#### 3. Job Submission Test
```
"Submit a model comparison job for examples/data/sample_rna.fa comparing Turner and Mix models"
```

#### 4. Job Management Test
```
"List all jobs and show me the status of the most recent one"
```

## Installing MXfold2 (Optional)

For full functionality, install the actual MXfold2 package:

```bash
# Activate the MCP environment
source env/bin/activate

# Clone and install MXfold2
git clone https://github.com/mxfold/mxfold2.git repo/mxfold2
cd repo/mxfold2
pip install -e .
cd ../..

# Test installation
python -c "import mxfold2; print('MXfold2 installed successfully')"
```

## Uninstalling

### Remove from Claude Code

```bash
claude mcp remove mxfold2
claude mcp list  # Verify removal
```

### Remove from Gemini CLI

Edit `~/.gemini/settings.json` and remove the `mxfold2` entry from `mcpServers`.

### Complete Removal

```bash
# Remove the entire MCP directory
rm -rf /path/to/mxfold2_mcp
```

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError: No module named 'fastmcp'"

```bash
# Install fastmcp in the environment
./env/bin/pip install fastmcp loguru
```

#### 2. "Permission denied" errors

```bash
# Fix permissions
chmod +x env/bin/python3.10
chmod +r src/server.py
chmod -R u+w jobs/
```

#### 3. "Server not responding"

```bash
# Test server manually
./env/bin/python src/server.py --help

# Check for port conflicts
netstat -tlnp | grep :6277
```

#### 4. "Tools not found"

```bash
# Verify tool count
./env/bin/python -c "
import sys; sys.path.insert(0, 'src')
with open('src/server.py') as f:
    content = f.read()
import re
print(f'Found {len(re.findall(r\"@mcp.tool\", content))} tools')
"
```

#### 5. Jobs fail with import errors

This is expected if MXfold2 isn't installed. The server will provide helpful error messages with installation instructions.

### Debug Mode

For debugging, start the server with verbose logging:

```bash
# Set debug environment
export MCP_LOG_LEVEL=DEBUG

# Run integration tests with debug
./env/bin/python tests/test_deployment_readiness.py
```

### Getting Help

1. **Check logs**: Look in `logs/` directory for server logs
2. **Check job logs**: Look in `jobs/[job_id]/job.log` for job-specific logs
3. **Run tests**: Use the provided test scripts to identify issues
4. **Verify paths**: Ensure all absolute paths are correct

## Advanced Configuration

### Custom Job Directory

```bash
# Create custom job directory
mkdir -p /custom/path/jobs

# Set environment variable
export MCP_JOBS_DIR="/custom/path/jobs"

# Test
./env/bin/python -c "
import sys; sys.path.insert(0, 'src/jobs')
from manager import job_manager
print(f'Jobs directory: {job_manager.jobs_dir}')
"
```

### Resource Limits

Edit `src/jobs/manager.py` to add resource limits:

```python
# Add timeout to subprocess calls
process = subprocess.Popen(
    cmd,
    stdout=log,
    stderr=subprocess.STDOUT,
    cwd=str(Path(script_path).parent.parent),
    timeout=3600  # 1 hour limit
)
```

### Custom Models

Add new models by editing the VALID_MODELS list in scripts:

```python
VALID_MODELS = ["Turner", "Mix", "MixC", "Zuker", "ZukerC", "ZukerL", "ZukerS", "CustomModel"]
```

## Production Deployment

### Security Considerations

1. **Isolate environment**: Run in a separate Python environment
2. **Limit resources**: Add memory and CPU limits
3. **Monitor jobs**: Set up monitoring for the jobs directory
4. **Log rotation**: Configure log rotation for production

### Performance Tuning

1. **Job concurrency**: Modify job manager for parallel execution
2. **Memory limits**: Add memory monitoring and limits
3. **Cleanup**: Set up automatic cleanup of old jobs

### Monitoring

Monitor these metrics in production:

- **Job success rate**: Track successful vs failed jobs
- **Execution times**: Monitor job execution duration
- **Disk usage**: Monitor jobs directory growth
- **Memory usage**: Track server memory consumption

## Support

For issues with:
- **MCP Protocol**: Check FastMCP documentation
- **Claude Code**: Check Claude Code documentation
- **MXfold2**: Check MXfold2 repository
- **This Server**: Check test files and logs in the installation directory