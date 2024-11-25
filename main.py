from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import openai
import os
from typing import List, Optional
import tempfile
import subprocess
import uvicorn
import threading
import time
import signal
import re
import pkg_resources
import ast
import sys
import socket
from github import Github, InputGitTreeElement
import github
from urllib.parse import quote
import requests
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

app.mount("/static", StaticFiles(directory="./"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("./index.html", "r") as f:
        return f.read()

# Configure OpenAI client
client = openai.OpenAI(
    api_key=os.environ.get("SAMBANOVA_API_KEY"),
    base_url="https://api.sambanova.ai/v1",
)

# Store running Streamlit processes
streamlit_processes = {}
streamlit_ports = {}
current_port = 8501

# Common Python packages and their pip equivalents
PACKAGE_MAPPING = {
    'PIL': 'pillow',
    'sklearn': 'scikit-learn',
    'cv2': 'opencv-python',
    'bs4': 'beautifulsoup4',
    'yaml': 'pyyaml',
    'PyPDF2': 'pypdf2',
    'pdf2': 'pypdf2'
}

# List of pre-approved packages that are safe to install
SAFE_PACKAGES = {
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'scikit-learn',
    'pillow', 'requests', 'beautifulsoup4', 'pyyaml', 'opencv-python',
    'plotly', 'altair', 'vega_datasets', 'pypdf2', 'python-docx',
    'openpyxl', 'xlrd', 'pdfplumber', 'pytesseract', 'wordcloud',
    'nltk', 'textblob', 'spacy', 'gensim', 'transformers', 'torch',
    'tensorflow', 'keras', 'sklearn', 'xgboost', 'lightgbm'
}

class PromptRequest(BaseModel):
    prompt: str

class MessageRequest(BaseModel):
    messages: List[dict]

class GitHubConfig(BaseModel):
    token: str = Field(..., description="GitHub personal access token")
    repo_name: Optional[str] = Field(None, description="Optional repository name")

class DeploymentRequest(BaseModel):
    code: str = Field(..., description="The Streamlit application code")
    github_config: GitHubConfig = Field(..., description="GitHub configuration")

    class Config:
        schema_extra = {
            "example": {
                "code": "import streamlit as st\n\nst.title('Hello World')",
                "github_config": {
                    "token": "ghp_xxxxxxxxxxxx",
                    "repo_name": "my-streamlit-app"
                }
            }
        }

class DependencyManager:
    @staticmethod
    def extract_imports(code: str) -> set:
        try:
            tree = ast.parse(code)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split('.')[0])
                        
            return imports
        except:
            import_lines = re.findall(r'^(?:from|import)\s+([a-zA-Z0-9_]+)', code, re.MULTILINE)
            return set(import_lines)

    @staticmethod
    def get_missing_packages(required_packages: set) -> set:
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        missing = set()
        
        for package in required_packages:
            pip_package = PACKAGE_MAPPING.get(package, package.lower())
            if pip_package not in installed_packages:
                missing.add(pip_package)
                
        return missing & SAFE_PACKAGES

    @staticmethod
    def install_packages(packages: set) -> bool:
        if not packages:
            return True
            
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'install', *packages],
                capture_output=True,
                text=True
            )
            print(f"Package installation output: {result.stdout}")
            print(f"Package installation errors: {result.stderr}")
            return result.returncode == 0
        except Exception as e:
            print(f"Error installing packages: {e}")
            return False

class DeploymentManager:
    def __init__(self, github_token: str):
        try:
            self.github = Github(github_token)
            self.user = self.github.get_user()
            # Verify token by trying to access user data
            self.username = self.user.login
        except Exception as e:
            raise Exception(f"Failed to authenticate with GitHub: {str(e)}")

    def create_repository(self, repo_name: str, code: str) -> dict:
        repo = None
        try:
            # Check if repo already exists
            try:
                existing_repo = self.user.get_repo(repo_name)
                raise Exception(f"Repository {repo_name} already exists")
            except github.GithubException as e:
                if e.status != 404:  # If error is not "Not Found"
                    raise

            print(f"Creating new repository: {repo_name}")
            # Create new repository
            repo = self.user.create_repo(
                name=repo_name,
                description="Streamlit application generated with AI",
                private=False,
                auto_init=False  # Changed to False to avoid conflicts
            )

            print("Repository created, initializing files...")
            
            # Create all necessary files
            files_to_create = {
                "README.md": f"""
# {repo_name}

Streamlit application generated with AI.

## Running Locally

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

## Deployment
This app can be deployed on [Streamlit Cloud](https://share.streamlit.io)
                """.strip(),
                
                "requirements.txt": """
streamlit>=1.24.0
pandas
numpy
pillow
                """.strip(),
                
                "app.py": code
            }

            # Create files one by one
            for file_name, content in files_to_create.items():
                try:
                    repo.create_file(
                        path=file_name,
                        message=f"Add {file_name}",
                        content=content,
                        branch="main"
                    )
                    print(f"Created {file_name}")
                except Exception as e:
                    print(f"Error creating {file_name}: {str(e)}")
                    raise

            return {
                "repo_url": repo.html_url,
                "clone_url": repo.clone_url,
                "deploy_url": f"https://share.streamlit.io/deploy?repository={quote(f'{self.username}/{repo_name}')}&branch=main"
            }

        except Exception as e:
            # Cleanup if repository was created but something failed
            if repo:
                try:
                    print(f"Error occurred, cleaning up repository {repo_name}")
                    repo.delete()
                except:
                    print(f"Failed to cleanup repository {repo_name}")
            raise Exception(f"Failed to create repository: {str(e)}")

def fix_streamlit_imports(code: str) -> str:
    # First clean up any malformed decorators
    code = re.sub(r'@+\s*st\.cache[_\w]*', '@st.cache_data', code)
    
    # Dictionary of old imports and their new versions
    import_fixes = {
        'from streamlit import caching': 'import streamlit as st',
        'streamlit.cache': 'st.cache_data',
        'st.cache': 'st.cache_data',
        '@st.cache': '@st.cache_data',
        'from streamlit import cache': 'import streamlit as st',
    }
    
    # Fix the imports
    modified_code = code
    for old, new in import_fixes.items():
        modified_code = modified_code.replace(old, new)
    
    # Add standard imports
    standard_imports = """
import streamlit as st
import pandas as pd
import io
import tempfile
"""
    if 'import streamlit' not in modified_code:
        modified_code = standard_imports + modified_code
        
    # Clean up decorator lines
    lines = modified_code.split('\n')
    final_lines = []
    for line in lines:
        if 'st.cache' in line:
            line = line.strip()
            if line.startswith('@'):
                line = '@st.cache_data'
            else:
                continue
        final_lines.append(line)
    
    return '\n'.join(final_lines)

def extract_python_code(text: str) -> str:
    code_blocks = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
    
    if code_blocks:
        code = code_blocks[0].strip()
    else:
        code = text.strip()
    
    code = code.strip()
    code = re.sub(r'^#\s+', '# ', code, flags=re.MULTILINE)
    code = re.sub(r'^\s*\*\s+', '', code, flags=re.MULTILINE)
    
    lines = code.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indent = len(line) - len(stripped)
            normalized_indent = (indent // 4) * 4
            cleaned_lines.append(' ' * normalized_indent + stripped)
        else:
            cleaned_lines.append('')
            
    fixed_code = fix_streamlit_imports('\n'.join(cleaned_lines))
    return fixed_code

def run_streamlit_app(file_path: str, port: int) -> bool:
    try:
        # Read the code
        with open(file_path, 'r') as f:
            code = f.read()
            
        # Add configuration for file upload
        config = """
import streamlit as st
import io
import tempfile

# Configure page
st.set_page_config(
    page_title="Generated App",
    layout="wide",
    initial_sidebar_state="collapsed"
)
"""
        # Combine configuration with the original code
        modified_code = config + code
        
        # Write the modified code back to the file
        with open(file_path, 'w') as f:
            f.write(modified_code)
            
        # Check and install dependencies
        dep_manager = DependencyManager()
        required_packages = dep_manager.extract_imports(modified_code)
        missing_packages = dep_manager.get_missing_packages(required_packages)
        
        if missing_packages:
            print(f"Installing missing packages: {missing_packages}")
            if not dep_manager.install_packages(missing_packages):
                print(f"Failed to install required packages: {missing_packages}")
                return False

        # Create a config file for Streamlit
        config_dir = os.path.join(tempfile.gettempdir(), f"streamlit_config_{port}")
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, ".streamlit", "config.toml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            f.write(f"""
[server]
port = {port}
address = "127.0.0.1"
enableCORS = true
enableXsrfProtection = false
maxUploadSize = 200

[browser]
serverAddress = "127.0.0.1"
serverPort = {port}
gatherUsageStats = false

[theme]
primaryColor = "#F63366"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
""")

        # Kill any existing process using this port
        if port in streamlit_processes:
            try:
                streamlit_processes[port].kill()
                time.sleep(0.5)
            except:
                pass

        # Set environment variables
        env = os.environ.copy()
        env.update({
            "STREAMLIT_CONFIG_DIR": os.path.dirname(config_path),
            "STREAMLIT_SERVER_PORT": str(port),
            "STREAMLIT_SERVER_ADDRESS": "127.0.0.1",
            "STREAMLIT_SERVER_ENABLE_CORS": "true",
            "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION": "false",
            "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
            "STREAMLIT_SERVER_HEADLESS": "true",
            "STREAMLIT_SERVER_RUN_ON_SAVE": "true",
            "STREAMLIT_BROWSER_FILE_WATCHER_TYPE": "none",
            "STREAMLIT_SERVER_COOKIE_SECRET": "dev",
            "STREAMLIT_SERVER_WEBSOCKET_COMPRESSION": "false",
        })

        # Start Streamlit process
        process = subprocess.Popen(
            ["streamlit", "run", file_path],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Quick check for immediate failure
        time.sleep(1)
        if process.poll() is not None:
            _, stderr = process.communicate()
            print(f"Streamlit failed to start: {stderr}")
            return False
            
        # Check port availability
        max_retries = 5
        retry_delay = 0.5
        
        for _ in range(max_retries):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex(('127.0.0.1', port))
                sock.close()
                
                if result == 0:
                    streamlit_processes[port] = process
                    print(f"Streamlit app running on http://127.0.0.1:{port}")
                    return True
                    
            except Exception as e:
                print(f"Retrying port check: {e}")
                
            time.sleep(retry_delay)
        
        process.kill()
        print(f"Timeout waiting for port {port}")
        return False
            
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        return False

@app.post("/generate-app")
async def generate_app(request: PromptRequest):
    global current_port
    try:
        system_prompt = """You are an AI expert at creating Streamlit applications. 
        Generate a complete, working Streamlit application based on the user's request.
        Important rules:
        1. For caching, ONLY use the @st.cache_data decorator
        2. Never use multiple cache decorators on the same function
        3. Always place cache decorators on their own line
        4. Never combine cache decorators with other decorators
        5. Always test data types before operations
        6. Include proper error handling
        Return ONLY the Python code, no explanations or markdown formatting."""
        
        response = client.chat.completions.create(
            model='Meta-Llama-3.1-8B-Instruct',
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt}
            ],
            temperature=0.1,
            top_p=0.1
        )
        
        raw_code = response.choices[0].message.content
        cleaned_code = extract_python_code(raw_code)
        
        if '@st.cache' in cleaned_code:
            cleaned_code = re.sub(r'@+st\.cache[_\w]*\s*\([^)]*\)', '@st.cache_data', cleaned_code)
            cleaned_code = re.sub(r'@+st\.cache[_\w]*\s*\n', '@st.cache_data\n', cleaned_code)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(cleaned_code)
            temp_file = f.name
        
        current_port += 1
        port = current_port
        
        success = run_streamlit_app(temp_file, port)
        
        if success:
            streamlit_ports[temp_file] = port
            return {
                "code": cleaned_code,
                "file_path": temp_file,
                "port": port,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start Streamlit app")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/iterate-app")
async def iterate_app(request: MessageRequest):
    try:
        response = client.chat.completions.create(
            model='Meta-Llama-3.1-8B-Instruct',
            messages=request.messages,
            temperature=0.1,
            top_p=0.1
        )
        
        raw_code = response.choices[0].message.content
        cleaned_code = extract_python_code(raw_code)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(cleaned_code)
            temp_file = f.name
            
        port = current_port + 1
        success = run_streamlit_app(temp_file, port)
        
        if success:
            streamlit_ports[temp_file] = port
            return {
                "response": cleaned_code,
                "port": port,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start Streamlit app")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/deploy")
async def deploy_app(request: DeploymentRequest):
    try:
        print("Starting deployment process...")
        
        if not request.code:
            raise HTTPException(status_code=400, detail="No code provided for deployment")
            
        if not request.github_config.token:
            raise HTTPException(status_code=400, detail="GitHub token is required")

        # Initialize deployment manager
        try:
            deployment_manager = DeploymentManager(request.github_config.token)
            print(f"Successfully authenticated as GitHub user: {deployment_manager.username}")
        except Exception as e:
            print(f"GitHub authentication error: {str(e)}")
            raise HTTPException(
                status_code=401, 
                detail=f"GitHub authentication failed: {str(e)}"
            )

        # Generate repository name if not provided
        repo_name = request.github_config.repo_name or f"streamlit-app-{int(time.time())}"
        
        # Validate repository name
        if not re.match(r'^[a-zA-Z0-9_-]+$', repo_name):
            raise HTTPException(
                status_code=400, 
                detail="Invalid repository name. Use only letters, numbers, hyphens, and underscores"
            )

        print(f"Creating repository: {repo_name}")
        
        try:
            result = deployment_manager.create_repository(repo_name, request.code)
            print(f"Repository created successfully at: {result['repo_url']}")
            
            return {
                "status": "success",
                "github_url": result["repo_url"],
                "deploy_url": result["deploy_url"],
                "message": "Repository created successfully. Click the deploy URL to complete deployment on Streamlit Cloud."
            }
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg:
                raise HTTPException(
                    status_code=400,
                    detail=f"Repository '{repo_name}' already exists. Please choose a different name."
                )
            else:
                print(f"Repository creation error: {error_msg}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to create repository: {error_msg}"
                )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected deployment error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during deployment: {str(e)}"
        )

@app.on_event("shutdown")
async def shutdown_event():
    for process in streamlit_processes.values():
        try:
            process.kill()
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)