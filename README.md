# 🧠 FUZ_AgenticAI

**Autonomous Code Modification & PR Agent powered by GPT + LangGraph**

An Agentic AI system that autonomously understands codebases, makes changes, runs tests, and opens pull requests on GitHub using GPT as the LLM, Pinecone for memory, and LangGraph for orchestration.

This project is designed to behave like a real software engineer — planning, coding, testing, reviewing, and fixing itself based on CI/CD feedback.

## 🚀 Key Capabilities

- 🤖 **Multi-agent architecture** (Planner, Coder, Tester, Reviewer)
- 🧩 **Stateful orchestration** using LangGraph
- 🧠 **Long-term memory** using Pinecone
- 🔧 **Tool usage** via GitHub API, Git CLI, and Shell
- 🔁 **CI/CD feedback loop** with automatic retries
- 🔐 **Safe GitHub workflows** (PR-based changes)

## 🏗️ System Architecture

```
User Request
     ↓
Orchestrator (LangGraph)
     ↓
Planner Agent (GPT-4 / GPT-4.1)
     ↓
┌───────────────┬────────────────┬───────────────┐
│ Code Agent    │ Test Agent     │ Review Agent  │
│ (LLM)         │ (Shell + CI)   │ (LLM)         │
└───────────────┴────────────────┴───────────────┘
     ↓
GitHub Agent (API + Git CLI)
     ↓
Commit → Push → Pull Request
     ↓
CI/CD Feedback
     ↓
Agent Fix Loop (on failure)
```

## 🧠 Agents Overview

### 1️⃣ Planner Agent (GPT)
- Understands the user request
- Analyzes repository context
- Breaks tasks into actionable steps
- Decides which agents to invoke

### 2️⃣ Code Agent (GPT)
- Modifies or creates code files
- Follows repository conventions
- Uses repository context from Pinecone
- Produces minimal, scoped diffs

### 3️⃣ Test Agent (Shell + CI)
- Runs tests locally via shell
- Triggers CI pipelines
- Collects test and build results

### 4️⃣ Review Agent (GPT)
- Reviews generated code
- Checks for bugs, security issues, and style violations
- Approves or requests fixes

### 5️⃣ GitHub Agent
- Manages Git operations
- Creates branches
- Commits changes
- Pushes code
- Opens Pull Requests
- Reads CI/CD statuses

## 🧠 Memory Layer (Pinecone)

Pinecone is used for long-term memory, including:
- Code embeddings
- Past commits and PRs
- Architectural decisions
- Error patterns and fixes

### Memory Types
- **Short-term**: Agent state (LangGraph)
- **Long-term**: Vectorized repo & decisions (Pinecone)

## 🔧 Tooling

| Tool | Purpose |
|------|---------|
| GPT-4 / GPT-4.1 | Reasoning, planning, coding, reviewing |
| LangGraph | Stateful agent orchestration |
| Pinecone | Long-term vector memory |
| GitHub API | Repo, PRs, CI status |
| Git CLI | Commit, diff, branch management |
| Shell | Tests, builds, linters |
| GitHub Actions | CI/CD execution |

## 🔁 CI/CD Fix Loop

1. Agent opens PR
2. CI pipeline runs
3. If ❌ failed:
   - Logs are fetched
   - Error context is sent back to agents
   - Code Agent applies fixes
   - New commit is pushed
   - Loop continues until ✅ success

## 📁 Project Structure

```
FUZ_AgenticAI/
│
├── agents/
│   ├── __init__.py
│   ├── planner.py
│   ├── coder.py
│   ├── tester.py
│   └── reviewer.py
│
├── orchestration/
│   └── graph.py
│
├── tools/
│   ├── __init__.py
│   ├── github_tool.py
│   ├── shell_tool.py
│   └── ci_tool.py
│
├── memory/
│   └── pinecone_store.py
│
├── config/
│   ├── llm.py
│   └── settings.py
│
├── main.py
├── requirements.txt
├── .env.example
└── README.md
```

## 🔐 Security & Best Practices

- ✅ PR-based changes (no direct main pushes)
- ✅ Scoped GitHub tokens
- ✅ Branch protection rules
- ✅ Command allow-list for shell execution
- ✅ LLM output validation

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd FUZ_AgenticAI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4

# Pinecone Configuration
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east1-gcp
PINECONE_INDEX_NAME=fuz-agentic-ai

# GitHub Configuration
GITHUB_TOKEN=your_github_token_here
GITHUB_REPO=owner/repo-name
GITHUB_BASE_BRANCH=main

# Agent Configuration
MAX_ITERATIONS=10
ENABLE_AUTO_FIX=true
```

### 4. Initialize Pinecone Index

The Pinecone index will be automatically created on first run if it doesn't exist.

### 5. Run the Agent

```bash
python main.py "Your request here"
```

Or run interactively:

```bash
python main.py
```

### 6. Command Line Options

```bash
python main.py "Add pagination to API endpoints" --max-iterations 15
python main.py "Fix failing tests" --no-auto-fix
```

## 📌 Example Use Cases

- "Refactor authentication logic to use JWT"
- "Fix failing CI tests"
- "Add pagination to API endpoints"
- "Upgrade dependency versions safely"
- "Apply linting fixes across repo"

## 🧭 Workflow Example

1. **User Request**: "Add pagination to the user list API endpoint"

2. **Planner Agent**: 
   - Analyzes the request
   - Identifies files to modify
   - Creates execution plan

3. **Code Agent**:
   - Reads existing API code
   - Generates pagination implementation
   - Writes changes to files

4. **Review Agent**:
   - Reviews code for quality
   - Checks for bugs and style issues

5. **Test Agent**:
   - Runs unit tests
   - Runs integration tests

6. **GitHub Agent**:
   - Creates branch
   - Commits changes
   - Opens PR

7. **CI/CD**:
   - Runs automated checks
   - If failed, agent fixes and retries

## 🧭 Roadmap

- [ ] Multi-repo support
- [ ] Web UI dashboard
- [ ] Policy-based code approval
- [ ] Slack / Discord integration
- [ ] Fine-grained agent permissions

## 🤝 Contributing

Contributions are welcome! Please open an issue or PR with clear context.

## 📜 License

MIT License

## 🙏 Acknowledgments

Built with:
- [OpenAI GPT](https://openai.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Pinecone](https://www.pinecone.io/)
- [PyGithub](https://github.com/PyGithub/PyGithub)

---

**Note**: This is an autonomous agent system. Always review PRs before merging to production.
