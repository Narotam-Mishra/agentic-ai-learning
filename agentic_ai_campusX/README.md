
# [Agentic AI using LangGraph](https://chat.deepseek.com/share/m1qobvkz31i2sr1zzp)

## 01. Agentic AI using LangGraph (16:57)

## Vision & Goals

**Three Goals:**

1. **Simple & Beginner-friendly** - Anyone can learn to build agentic applications easily
2. **Strong command over LangGraph** - Deep fundamentals coverage
3. **Conceptual depth** - If LangGraph gets replaced, you can learn new frameworks easily

---

## Curriculum (6 Modules)

| Module | Topic | What You'll Learn |
|--------|-------|-------------------|
| **1** | Foundations of Agentic AI | Agentic AI vs AI Agents vs Generative AI, Agentic RAG vs Traditional RAG, Top frameworks |
| **2** | LangGraph Fundamentals | Graph building, State concept, Nodes, Edges, Conditional edges, Looping, Popular AI workflows |
| **3** | Advanced LangGraph | Persistence, Memory, Human-in-the-loop, Breakpoints, Checkpointers, Time travel |
| **4** | AI Agents Building | ReAct Agent, Reflection pattern, Self-ASK pattern, Planning, Multi-agent systems |
| **5** | Agentic RAG Applications | Self-RAG, Corrective RAG (CRAG), Advanced RAG architectures |
| **6** | Production | UI, Debugging, Observability, LangSmith integration, Deployment |

**Total videos:** Estimated 35-50 videos

**Upload frequency:** 3 videos per week (target)

---

## Prerequisites

| # | Requirement | Level |
|---|-------------|-------|
| 1 | **Python** | Intermediate (OOP, Typing module, Pydantic, Asyncio IO) |
| 2 | **LLMs** | Familiarity with working on LLMs |
| 3 | **LangChain** | Required - LangGraph is built on LangChain |

> ⚠️ Highly recommended to watch his LangChain playlist (18 videos) first.

---

## Important Concepts with Basic Code Examples

### 1. What is Agentic AI?

**Simple definition:** AI systems that can act autonomously, make decisions, and perform tasks without human intervention.

```python
# Simple concept: An agent that decides which tool to use
class SimpleAgent:
    def __init__(self, task):
        self.task = task
    
    def decide_action(self):
        if "calculate" in self.task:
            return "use_calculator"
        elif "search" in self.task:
            return "use_search"
        else:
            return "use_llm"
```

### 2. LangGraph - Graph, Nodes, Edges, State

**Node** = A step/function that processes information
**Edge** = Connection between nodes (where to go next)
**State** = Shared data container that passes between nodes

```python
from typing import TypedDict

# 1. Define State (shared data)
class AgentState(TypedDict):
    messages: list
    current_step: str

# 2. Define Nodes (functions)
def process_node(state: AgentState):
    state["messages"].append("Processing...")
    state["current_step"] = "done"
    return state

def validate_node(state: AgentState):
    if "error" in state["messages"]:
        return "error_handler"
    return "success"

# 3. Graph structure concept
"""
    START → process_node → validate_node → 
        ├─ (if error) → error_handler → END
        └─ (if success) → END
"""
```

### 3. Conditional Edges

Routing to different nodes based on conditions:

```python
# Conditional edge logic
def route_condition(state: AgentState) -> str:
    if state["current_step"] == "complete":
        return "end_node"
    elif state["needs_more_info"]:
        return "retrieval_node"
    else:
        return "processing_node"

# Usage: After a node, based on condition, go to different nodes
```

### 4. Looping in LangGraph

Repeating nodes until a condition is met:

```python
# Loop concept - iterative refinement
class LoopExample:
    def __init__(self, max_iterations=5):
        self.max_iterations = max_iterations
        self.iteration = 0
    
    def should_continue(self, state):
        if self.iteration >= self.max_iterations:
            return "end"
        if state["answer_quality"] > 0.9:
            return "end"
        self.iteration += 1
        return "continue"  # Goes back to processing node
```

### 5. AI Agent Design Patterns

**ReAct Pattern (Reason + Act):**

```python
# ReAct agent concept
class ReActAgent:
    def run(self, question):
        # 1. REASON - Think about what to do
        thought = self.think(question)
        
        # 2. ACT - Take action using a tool
        if thought["action"] == "search":
            result = self.search(thought["query"])
        elif thought["action"] == "calculate":
            result = self.calculate(thought["query"])
        
        # 3. OBSERVE - Get result and decide next
        observation = self.observe(result)
        
        # Loop until final answer
        return self.generate_answer(observation)
```

**Reflection Pattern:**

```python
# Agent reflects on its own output
class ReflectionAgent:
    def generate_response(self, query):
        # First attempt
        draft = self.llm.generate(query)
        
        # Reflect on the draft
        reflection = self.llm.reflect(f"Is this good? {draft}")
        
        # Improve based on reflection
        if reflection["needs_improvement"]:
            improved = self.llm.generate(f"Improve this: {draft}\nFeedback: {reflection}")
            return improved
        return draft
```

**Multi-Agent System:**

```python
# Multiple agents collaborating
class MultiAgentSystem:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.writer = WriterAgent()
        self.reviewer = ReviewAgent()
    
    def work_together(self, topic):
        research = self.researcher.research(topic)
        draft = self.writer.write(research)
        feedback = self.reviewer.review(draft)
        
        if feedback["approved"]:
            return draft
        else:
            return self.writer.revise(draft, feedback)
```

### 6. Memory & Persistence

```python
# Checkpointer concept - saving state
class CheckpointExample:
    def save_state(self, state, checkpoint_id):
        # Save current execution state
        self.storage[checkpoint_id] = state.copy()
    
    def load_state(self, checkpoint_id):
        # Resume from saved state
        return self.storage[checkpoint_id]
    
    def time_travel(self, target_checkpoint):
        # Go back to any previous state
        return self.load_state(target_checkpoint)
```

### 7. Human-in-the-Loop

```python
class HumanInLoop:
    def run_with_approval(self, action):
        # Execute action
        result = self.execute(action)
        
        # Ask human for approval
        print(f"Action result: {result}")
        human_input = input("Approve? (y/n): ")
        
        if human_input == 'y':
            return self.continue_execution()
        else:
            return self.rollback_and_revise()
```

### 8. Agentic RAG (vs Traditional RAG)

**Traditional RAG:** Retrieve → Augment → Generate (one fixed path)
**Agentic RAG:** Agent decides WHEN and WHAT to retrieve

```python
# Traditional RAG
def traditional_rag(query):
    docs = vector_search(query)  # Always searches
    return llm.generate(query, docs)

# Agentic RAG
def agentic_rag(query, agent):
    # Agent decides if retrieval is needed
    decision = agent.decide(query)
    
    if decision == "need_retrieval":
        docs = vector_search(query)
        return llm.generate(query, docs)
    elif decision == "need_multiple_searches":
        docs1 = vector_search(query)
        docs2 = web_search(query)
        return llm.generate(query, docs1 + docs2)
    else:
        return llm.generate(query)  # No retrieval needed
```

### 9. Production Concepts

```python
# Basic observability setup
import logging
from langsmith import Client

class ProductionAgent:
    def __init__(self):
        self.logger = logging.getLogger("agent")
        self.langsmith = Client()
    
    def run_with_tracing(self, input_data):
        # Log every step
        self.logger.info(f"Starting with input: {input_data}")
        
        # Trace with LangSmith
        with self.langsmith.trace("agent_execution") as trace:
            result = self.process(input_data)
            trace.add_output(result)
        
        self.logger.info(f"Completed: {result}")
        return result
```

---

## Key Takeaways

1. **Start with prerequisites** - Intermediate Python + LLM familiarity + LangChain
2. **Don't skip fundamentals** - Module 1 & 2 are crucial
3. **Follow sequentially** - Each module builds on previous
4. **Expect changes** - This space evolves rapidly, curriculum will adapt
5. **Production focus** - Final module covers real-world deployment

---

## 02. Generative AI vs Agentic AI (01:02:43)

This lecture explains the key differences between Generative AI and Agentic AI by walking through a real-world **HR hiring scenario**. The instructor shows how a simple chatbot evolves through 4 stages to become a fully autonomous AI agent.

---

## Important Pointers

| # | Key Takeaway |
|---|--------------|
| 1 | **Generative AI** focuses on **creating content** (text, images, video, code, speech) |
| 2 | **Agentic AI** focuses on **achieving goals** autonomously through planning and action |
| 3 | GenAI is **reactive** → needs step-by-step human guidance |
| 4 | Agentic AI is **proactive** → given a goal, it plans and executes autonomously |
| 5 | Generative AI is a **building block** / subset of Agentic AI |
| 6 | Agentic AI combines: **Planning + Reasoning + Memory + Tools + LLMs** |
| 7 | Traditional AI learns patterns between input/output; GenAI learns the **distribution of data** |
| 8 | GenAI output feels like **human-created** content |

---

## What is Generative AI?

**Simple definition:** A class of AI models that can create **new content** (text, images, audio, code, video) that resembles human-created data.

**How it works (vs Traditional AI):**
- Traditional AI: Learns relationship between input → output (e.g., classify spam)
- GenAI: Learns the **entire distribution/nature of data** → can generate **new samples** from that distribution

**Example:** Give a GenAI model many cat images → it learns "what a cat looks like in real life" → then generates a brand new cat image.

```python
# Conceptual: Traditional AI vs Generative AI

# Traditional AI - Classification
def traditional_ai_classify(email):
    # Learns pattern: specific words -> spam or not
    if "lottery" in email or "prize" in email:
        return "spam"
    return "not_spam"

# Generative AI - Content creation
def generative_ai_generate(prompt):
    # Learns distribution of human language
    # Then generates NEW text that feels human-written
    return model.generate(prompt)  # "Write a poem about cats"
```

**Popular GenAI applications:**
- Chatbots: ChatGPT, Gemini, Claude, Grok
- Image generation: DALL-E, Midjourney
- Code generation: Code Llama
- Text-to-speech: Eleven Labs
- Video generation: Sora, Runway

---

## The Evolution: 4 Chatbot Stages in Hiring Scenario

**Problem:** You're an HR recruiter needing to hire a Backend Engineer (2-4 years experience). Tasks: draft JD → post on job portals → shortlist candidates → schedule interviews → send offer → onboard.

### Stage 1: Simple LLM Chatbot (Basic GenAI)

**How it works:** You ask, chatbot answers. Reactive, generic advice, no memory, no actions.

```python
class SimpleLLMChatbot:
    def ask(self, user_query):
        if "draft JD" in user_query:
            return "Here's a generic JD: Looking for backend engineer with Python..."
        elif "interview questions" in user_query:
            return "Ask about Python, frameworks, problem-solving..."
        else:
            return "I can help with basic HR tasks."
```

**Problems:**
- ❌ **Reactive** – waits for your prompt
- ❌ **No memory** – forgets previous conversations
- ❌ **Generic advice** – not company-specific
- ❌ **Cannot take actions** – can't post jobs or send emails

---

### Stage 2: RAG-Based Chatbot (Context-Aware)

**RAG = Retrieval Augmented Generation**

Adds company knowledge base (past JDs, hiring playbooks, salary bands, interview question bank, offer letter templates).

```python
# Simple RAG concept
class RAGChatbot:
    def __init__(self, company_knowledge_base):
        self.kb = company_knowledge_base  # past JDs, salary data, etc.
    
    def answer(self, query):
        # 1. RETRIEVE relevant company documents
        relevant_docs = self.kb.search(query)
        
        # 2. AUGMENT prompt with company context
        enhanced_prompt = f"Company context: {relevant_docs}\nUser query: {query}"
        
        # 3. GENERATE response
        return self.llm.generate(enhanced_prompt)

# Usage
kb = CompanyKB(jd_templates=[...], salary_bands=[...], interview_qs=[...])
bot = RAGChatbot(kb)
response = bot.answer("Draft JD for backend engineer")
# Returns: Tailored JD with company-specific tech stack (Python, Django) and salary band
```

**Improvements:**
- ✅ **Specific advice** – tailored to company DNA
- ✅ Uses past hiring data

**Remaining problems:**
- ❌ Still reactive
- ❌ No memory
- ❌ Cannot take actions

---

### Stage 3: Tool-Augmented Chatbot

Adds integrations with external tools via APIs: LinkedIn API, resume parser, calendar API, email API, HRM software.

```python
class ToolAugmentedChatbot:
    def __init__(self, tools):
        self.tools = tools  # e.g., linkedin_api, email_api, calendar_api
    
    def execute(self, user_command):
        if "post job on LinkedIn" in user_command:
            # Tool use: Actually posts the job
            return self.tools.linkedin_api.post_job(jd)
        elif "schedule interview" in user_command:
            # Tool use: Checks calendar and sends invites
            free_slots = self.tools.calendar_api.get_free_slots()
            self.tools.email_api.send_invite(candidate, free_slots[0])
            return "Interview scheduled"
        elif "shortlist candidates" in user_command:
            # Tool use: Downloads and parses resumes
            resumes = self.tools.linkedin_api.get_resumes()
            parsed = self.tools.resume_parser.parse(resumes)
            return self.rank_candidates(parsed)

# Example tools
class LinkedInAPI:
    def post_job(self, jd): return "Job posted"
    def boost_post(self, post_id): return "Post boosted"

class CalendarAPI:
    def get_free_slots(self): return ["Friday 10 AM", "Friday 2 PM"]

class EmailAPI:
    def send_email(self, to, subject, body): return "Email sent"
```

**Improvements:**
- ✅ Can **take actions** (post jobs, send emails, schedule interviews)
- ✅ Still has RAG (company-specific)

**Remaining problems:**
- ❌ Still reactive
- ❌ No memory / context awareness
- ❌ Cannot adapt when things go wrong (e.g., low applications)

---

### Stage 4: Agentic AI (The Goal)

**Definition:** An AI system that is **proactive, context-aware, and adaptable** – given a goal, it autonomously plans and executes steps, involving humans only for approval.

```python
# Simplified Agentic AI concept
class AgenticAI:
    def __init__(self, tools, knowledge_base, memory):
        self.tools = tools
        self.kb = knowledge_base
        self.memory = memory  # remembers past steps
        self.plans = []
    
    def run(self, goal):
        # 1. UNDERSTAND goal
        # 2. PLAN steps
        self.plans = self.create_plan(goal)
        # 3. EXECUTE each step autonomously
        for step in self.plans:
            result = self.execute_step(step)
            self.memory.remember(step, result)
            # 4. MONITOR and ADAPT if problems arise
            if self.has_problem(result):
                self.adapt_plan()
        return "Goal achieved"
    
    def create_plan(self, goal):
        # "Hire backend engineer" -> 
        # [draft_JD, post_JD, monitor_applications, shortlist, 
        #  schedule_interviews, conduct_interviews, send_offer, onboard]
        return self.llm.plan(goal, self.kb)
    
    def execute_step(self, step):
        if step == "post_JD":
            return self.tools.linkedin_api.post_job(self.jd)
        elif step == "monitor_applications":
            count = self.tools.linkedin_api.get_application_count()
            if count < 5:  # ADAPTATION
                self.suggest_boosting()
            return count
```

**Key capabilities demonstrated in the video:**

1. **Proactive** – After receiving "Hire a backend engineer", the agent automatically:
   - Drafts JD using company data
   - Posts to LinkedIn & Naukri
   - Monitors application counts
   - Notifies when applications are low
   - Suggests solutions (broaden JD, boost posts)
   - Gets human approval, then executes changes

2. **Context-aware (Memory)** – Remembers:
   - That it already drafted a JD
   - Which platforms it posted to
   - Which candidates were shortlisted
   - That interviews are scheduled for Friday

3. **Adaptable** – When only 2 applications came (below expectation), the agent:
   - Detected the problem autonomously
   - Suggested: "Broaden JD to include Full Stack" + "Boost LinkedIn post"
   - Waited for human approval, then executed both

4. **Tool usage** – Posts jobs, checks calendar, sends emails, parses resumes, triggers onboarding

5. **Human-in-the-loop** – Asks for approvals at key decision points, but does all heavy lifting

```python
# Practical example: Agentic hiring flow
class HiringAgent(AgenticAI):
    def run_hiring(self, role, experience_years):
        # Step 1: Draft JD
        jd = self.draft_jd(role, experience_years)
        self.request_approval("JD ready", jd)
        
        # Step 2: Post jobs
        self.post_to_platforms(jd, ["LinkedIn", "Naukri"])
        
        # Step 3: Monitor (proactive)
        while self.days_elapsed < 7:
            apps = self.check_applications()
            if apps < 5 and self.days_elapsed > 2:
                # Self-adaptation
                self.suggest_and_execute("Low applications. Boost posts?")
            time.sleep(86400)
        
        # Step 4: Auto-shortlist using resume parser
        candidates = self.parse_and_rank_resumes()
        self.notify_user(f"Top {len(candidates)} candidates identified")
        
        # Step 5: Auto-schedule interviews via calendar
        slots = self.get_free_slots()
        self.send_invites(candidates[:2], slots[0])
        
        # Step 6: After interview feedback, auto-send offer
        self.generate_and_send_offer(selected_candidate)
        
        # Step 7: Trigger onboarding
        self.trigger_onboarding(selected_candidate)
        return "Hiring complete"
```

---

## Final Comparison Table

| Feature | Generative AI | Agentic AI |
|---------|---------------|------------|
| **Primary focus** | Create content (text, images, etc.) | Achieve a goal |
| **Behavior** | Reactive – responds to prompts | Proactive – initiates actions |
| **Human involvement** | Guides every step | Only gives high-level goal + approvals |
| **Memory** | Typically none (stateless) | Has context memory |
| **Tool use** | No (can't act) | Yes (calls APIs, sends emails, etc.) |
| **Adaptability** | No | Yes – detects problems and adjusts plans |
| **Planning** | No | Yes – breaks goals into steps |
| **Example** | "Write a job description" | "Hire a backend engineer" → agent does everything |

---

## Key Quote from Video

> **"Generative AI is a capability, Agentic AI is a behavior."**

Agentic AI uses GenAI (LLMs) as its **brain** for reasoning and planning, but adds:
- Memory
- Tool use
- Planning & reasoning loops
- Adaptability
- Autonomy

---

## 03. What is Agentic AI? (01:00:24)

summaries this agentic ai tutorial transcript in simple words with all detail, make note of all important pointers and also explain each important concepts with basic code examples