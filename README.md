# 🛰️ Autonomous Agent Agency (3-Tier Architecture)

This repository provides the blueprints for a 24/7 Autonomous Intelligence Agency running locally, bridging the gap between raw research and mobile execution.

## 🏗️ The Architecture
1. **The Brain:** Gemini CLI + Gemma 4 (31B) / Mistral 7B via OpenRouter.
2. **The Hands:** MCP (Model Context Protocol) for live web browsing and GitHub management.
3. **The Bridge:** Gmail (Command Input) + WhatsApp (Briefing Output via TextMeBot).

## 🚀 Core Automated Workflows

### 1. Daily AI Intelligence Briefing
Scans the ArXiv API (`cat:cs.AI`) every morning, fetches the latest 5 papers, and uses Gemma 4 to perform an architectural autopsy.
- **Output:** A WhatsApp briefing covering: Architecture, X-Factor, and Strategic Value.

### 2. WhatsApp Remote Control
A "Command-Line-via-Email" system.
- **Input:** Email with subject `CMD: [Your Task]`.
- **Logic:** Agent processes the task using live web search (MCP).
- **Output:** Final report delivered to WhatsApp.

## 🛠️ Technical Setup

### Prerequisites
- **Python 3.10+**
- **Gemini CLI** (configured with MCP)
- **OpenRouter API Key** (for Gemma 4 access)
- **TextMeBot API Key** (for WhatsApp bridge)

### Installation
1. **Clone the Agency:**
   ```bash
   git clone https://github.com/almonzeir/Autonomous-Agent-Agency.git
   cd Autonomous-Agent-Agency
   ```
2. **Configure Environment:**
   Create a `.env` file with your keys for Gmail (App Password), OpenRouter, and TextMeBot.

3. **Schedule the Agency:**
   Use Windows Task Scheduler (or `cron` on Linux) to run the `run_agency.bat` daily.

---
*Developed and documented autonomously by the Almonzer Hamid AI Agentic System via Gemini CLI.*
