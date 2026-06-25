
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

This lecture provides a **formal definition** of Agentic AI, its key characteristics, and how it differs from reactive systems like chatbots. The instructor uses the same HR hiring example from the previous video to illustrate each concept.

---

## Important Pointers

| # | Key Takeaway |
|---|--------------|
| 1 | **Agentic AI** = system that takes a goal from user and works toward completing it **autonomously** (minimal human guidance) |
| 2 | Agentic AI is **proactive**, not reactive – it initiates actions and plans on its own |
| 3 | 6 key characteristics: **Autonomy, Goal-Oriented, Planning, Reasoning, Adaptability, Context Awareness** |
| 4 | Autonomy can be **controlled** via permissions, human-in-the-loop, overrides, and guardrails |
| 5 | Planning = breaking a high-level goal into structured sequence of sub-goals |
| 6 | Planning involves: generate multiple candidate plans → evaluate → select best |
| 7 | Reasoning is needed in **both planning and execution** stages |
| 8 | Adaptability = ability to modify plans when unexpected conditions occur |
| 9 | Context awareness = retaining and utilizing information from ongoing tasks, past interactions, environment |
| 10 | Two types of memory: **Short-term** (current session) and **Long-term** (persistent rules/preferences) |

---

## What is Agentic AI? (Simple Definition)

> **Agentic AI** is a type of AI that takes a task/goal from a user and then works toward completing it on its own, with minimal human guidance. It plans, takes actions, adapts to changes, and seeks help only when necessary.

**Contrast with Generative AI (ChatGPT-style):**
- **Reactive:** You ask a question → chatbot answers. You guide every step.
- **Agentic:** You give a goal ("Plan my Goa trip") → agent autonomously figures out dates, transport, hotels, itinerary, and presents complete plan.

---

## The HR Hiring Example (Brief)

**Goal given to Agentic AI:** "Hire a remote backend engineer with 2-4 years experience"

The agent autonomously:
1. Drafts job description using company docs
2. Posts to LinkedIn & Naukri via APIs
3. Monitors applications daily
4. Detects low applications → suggests solutions (broaden JD, boost ads) → gets approval → executes
5. Shortlists candidates using resume parser
6. Checks calendar & schedules interviews
7. Sends offer letter and triggers onboarding

**Human only gives approval at key decision points.**

---

## 6 Key Characteristics of Agentic AI Systems

### 1. Autonomy

**Definition:** Ability to make decisions and take actions on its own to achieve a goal, without step-by-step human instructions.

```python
# Reactive chatbot (NOT autonomous)
def reactive_chatbot():
    while True:
        user_input = input("What do you need? ")
        if "draft JD" in user_input:
            print(draft_jd())
        elif "post job" in user_input:
            print("You need to post it manually")

# Agentic AI (autonomous)
class AutonomousAgent:
    def __init__(self, goal):
        self.goal = goal
        self.plan = []
    
    def run(self):
        # Autonomously plans and executes
        self.plan = self.create_plan(self.goal)
        for step in self.plan:
            self.execute_step(step)  # No human asking each time
        return "Goal achieved"
```

**Controlling Autonomy (Important!):**
- **Limit tools/actions** – restrict what agent can do independently
- **Human-in-the-loop** – require approval before critical actions
- **Override controls** – pause/stop agent anytime
- **Guardrails** – hard rules (e.g., "never schedule weekends")

```python
class ControlledAgent:
    def execute_with_approval(self, action, critical=False):
        if critical:
            approval = input(f"Approve {action}? (y/n): ")
            if approval != 'y':
                return "Action blocked"
        return self.do(action)
    
    # Guardrail: hard-coded rule
    def schedule_interview(self, date):
        if date.weekday() >= 5:  # Saturday or Sunday
            raise ValueError("Weekend interviews not allowed")
        return self.calendar.schedule(date)
```

---

### 2. Goal-Oriented

**Definition:** Operates with a persistent objective in mind and continuously directs actions to achieve that objective, rather than just responding to isolated prompts.

```python
class GoalOrientedAgent:
    def __init__(self):
        self.goal = None
        self.progress = {}
    
    def set_goal(self, goal_description, constraints=None):
        self.goal = {
            "main_goal": goal_description,
            "constraints": constraints or [],
            "status": "active",
            "created_at": timestamp(),
            "progress": {
                "completed_tasks": [],
                "pending_tasks": []
            }
        }
    
    def check_progress(self):
        # Returns: "JD drafted", "Posted on LinkedIn", etc.
        return self.goal["progress"]
    
    def is_goal_complete(self):
        return self.goal["status"] == "completed"

# Example goal structure (JSON-like)
goal = {
    "main_goal": "Hire a backend engineer",
    "constraints": [
        "2-4 years experience",
        "Remote only",
        "Budget under $5000"
    ],
    "status": "active",
    "progress": {
        "JD drafted": True,
        "Posted on LinkedIn": True,
        "Applications received": 8,
        "Interviews scheduled": 2
    }
}
```

**Goals can be altered mid-way** – agent re-plans automatically.

---

### 3. Planning

**Definition:** Ability to break down a high-level goal into a structured sequence of actions and sub-goals.

**The 3-step planning process:**

```python
class PlanningAgent:
    def plan(self, goal):
        # STEP 1: Generate multiple candidate plans
        candidate_plans = [
            ["draft_JD", "post_on_LinkedIn", "monitor", "shortlist", "interview", "offer"],
            ["draft_JD", "use_referral_program", "hire_agency", "interview"],
            ["find_freelancer", "negotiate", "onboard"]
        ]
        
        # STEP 2: Evaluate each plan
        evaluated_plans = []
        for plan in candidate_plans:
            score = self.evaluate_plan(plan, criteria=[
                "efficiency",      # How fast?
                "tool_availability", # Do we have required APIs?
                "cost",            # Within budget?
                "risk",            # Likelihood of failure?
                "constraint_alignment" # Remote-only? etc.
            ])
            evaluated_plans.append((plan, score))
        
        # STEP 3: Select the best plan
        best_plan = max(evaluated_plans, key=lambda x: x[1])
        return best_plan[0]
    
    def execute_plan(self, plan):
        # Executes iteratively – can re-plan if a step fails
        for step in plan:
            success = self.execute_step(step)
            if not success:
                return self.replan_and_continue(step)
        return "Plan completed"
```

**Key insight:** Planning is an **iterative loop** – if execution fails at step 4, agent goes back to planning stage and creates a new plan.

---

### 4. Reasoning

**Definition:** The cognitive process through which an agent interprets information, draws conclusions, and makes decisions (during both planning and execution).

```python
class ReasoningAgent:
    def reason_about_planning(self, goal):
        # Goal decomposition reasoning
        if "hire" in goal:
            # Reason: Hiring requires JD, posting, screening, etc.
            return ["draft_JD", "post_job", "screen_candidates"]
        
        # Tool selection reasoning
        if step == "find_salary_range":
            # Reason: I need external data → use search tool
            return self.tools.search
        
        # Resource estimation reasoning
        if complexity == "high":
            return {"estimated_days": 7, "risks": ["low applications"]}
    
    def reason_during_execution(self, observation):
        # Decision making: which candidate to shortlist?
        if observation["match_score"] > 0.8:
            return "shortlist"
        elif observation["match_score"] > 0.5:
            return "maybe_ask_human"
        else:
            return "reject"
        
        # Human-in-the-loop reasoning
        if self.confidence < 0.6:
            return self.ask_human_for_help()
        
        # Error handling reasoning
        if self.tool_available(api_name) == False:
            # Reason: Tool down → find alternative
            return self.find_alternative_tool()
```

**Example from video:** Agent notices low applications (observation) → reasons that JD might be too narrow or post not promoted → decides to suggest broadening JD and boosting ads.

---

### 5. Adaptability

**Definition:** Ability to modify plans, strategies, and actions in response to unexpected conditions, while staying aligned with the goal.

```python
class AdaptableAgent:
    def execute_with_adaptation(self, plan):
        for step in plan:
            try:
                result = self.execute_step(step)
                self.monitor(result)
            except ToolFailureError:
                # ADAPT: Tool failed
                self.adapt_to_tool_failure(step)
            except ExternalFeedback as feedback:
                # ADAPT: Environment says something's wrong
                if feedback.low_applications:
                    self.adapt_strategy({"broaden_JD": True, "boost_ads": True})
            except GoalChangedError:
                # ADAPT: Goal changed mid-way
                return self.replan_with_new_goal()
    
    def adapt_to_tool_failure(self, failed_step):
        # Example: Calendar API down
        if failed_step.tool == "calendar_api":
            # Instead of failing, ask human directly
            availability = self.ask_human("When are you free?")
            return self.schedule_manually(availability)
    
    def adapt_strategy(self, changes):
        # Example: Low applications
        if changes.get("broaden_JD"):
            self.jd["title"] = "Backend/Full Stack Engineer"
            self.repost_job()
        if changes.get("boost_ads"):
            self.linkedin_api.boost_post(self.job_id, budget=100)
```

**Three triggers for adaptation:**
1. **Tool failures** – API down, timeout, etc.
2. **External feedback** – low applications, candidate rejection
3. **Goal changes** – user changes requirements mid-process

---

### 6. Context Awareness

**Definition:** Ability to understand, retain, and utilize relevant information from ongoing tasks, past interactions, user preferences, and environmental cues.

```python
class ContextAwareAgent:
    def __init__(self):
        self.short_term_memory = {}   # Current session
        self.long_term_memory = {}    # Persistent rules/preferences
    
    def store_context(self, key, value, persistent=False):
        if persistent:
            self.long_term_memory[key] = value
        else:
            self.short_term_memory[key] = value
    
    def get_context(self, key):
        # Check short-term first, then long-term
        if key in self.short_term_memory:
            return self.short_term_memory[key]
        return self.long_term_memory.get(key)
    
    def run_hiring(self):
        # Stores original goal
        self.store_context("goal", "Hire backend engineer", persistent=True)
        
        # Stores progress
        self.store_context("progress", {"JD_drafted": True, "posted": False})
        
        # Stores environment state
        self.store_context("environment", {
            "linkedin_job_id": "12345",
            "applications_received": 8,
            "ad_budget_remaining": 50
        })
        
        # Stores tool responses
        resume_data = self.resume_parser.parse(candidate_resume)
        self.store_context(f"candidate_{id}", resume_data)
        
        # Stores user preferences (persistent)
        self.store_context("prefers_remote", True, persistent=True)
        self.store_context("max_budget", 5000, persistent=True)
        
        # Stores guardrails (persistent)
        self.store_context("guardrail", "no_offer_without_approval", persistent=True)
```

**Types of Context Stored:**

| Context Type | Example | Memory Type |
|--------------|---------|-------------|
| Original goal | "Hire backend engineer" | Long-term |
| Progress | JD drafted, 8 applications | Short-term |
| Environment | LinkedIn job ID, ad budget | Short-term |
| Tool responses | "Candidate has 3 years Django" | Short-term |
| User preferences | Remote-only, budget limit | Long-term |
| Guardrails | "No offers without approval" | Long-term |

---

## Two Types of Memory in Agentic AI

```python
class AgentMemory:
    def __init__(self):
        # Short-term: current session only
        self.short_term = {
            "current_goal": "Hire backend engineer",
            "completed_steps": ["draft_JD", "post_on_LinkedIn"],
            "tool_responses": {"resume_parser": "candidate_A_score: 0.9"}
        }
        
        # Long-term: persists across sessions
        self.long_term = {
            "user_preferences": {"remote_only": True, "max_budget": 5000},
            "guardrails": ["no_weekend_interviews", "approval_required_for_offer"],
            "company_policies": {"salary_band_2_4_years": "8-12 LPA"}
        }
    
    def recall(self):
        # Agent uses both to make decisions
        if self.long_term["guardrails"]["no_weekend_interviews"]:
            self.schedule_filter.avoid_weekends()
        
        if self.short_term["completed_steps"]:
            next_step = self.get_next_step()
            return next_step
```

---

## Summary Table: Agentic AI vs Generative AI (Chatbots)

| Feature | Generative AI (ChatGPT) | Agentic AI |
|---------|------------------------|-------------|
| **Interaction style** | Reactive – responds to prompts | Proactive – initiates actions |
| **Human involvement** | Guides every step | Only gives goal + occasional approvals |
| **Memory** | None (stateless) | Short-term + long-term memory |
| **Planning** | No | Yes – breaks goals into steps |
| **Reasoning** | Limited | Extensive (planning + execution) |
| **Tool use** | No | Yes (APIs, search, calendar, email) |
| **Adaptability** | No | Yes – handles failures and changes |
| **Example** | "Write a JD" | "Hire a backend engineer" → agent does everything |

---

## Key Pointers

> **"Autonomy + Goal Orientation + Planning + Reasoning + Adaptability + Context Awareness = Agentic AI"**

Generative AI is a **capability** (content creation). Agentic AI is a **behavior** (autonomous goal achievement). Agentic AI uses GenAI (LLMs) as its brain for reasoning and planning, but adds memory, tools, and adaptability.

---

This section covers the **5 core components** found in almost every Agentic AI application. The instructor explains each component's role and how they work together.

## Important Pointers

| # | Component | Role |
|---|-----------|------|
| 1 | **Brain** | The LLM – interprets goals, plans, reasons, selects tools, communicates |
| 2 | **Orchestrator** | Executes the plan – sequences tasks, conditional routing, retries, looping, delegation |
| 3 | **Tools** | Hands and legs of the agent – interact with external world (APIs, databases, email, search, RAG) |
| 4 | **Memory** | Stores short-term (session) and long-term (persistent) information, tracks state |
| 5 | **Supervisor** | Implements human-in-the-loop – approvals, guardrails, escalations |

---

## 1. Brain (The LLM)

**Definition:** The core intelligence – typically an LLM (Large Language Model) that handles all heavy cognitive lifting.

**What the Brain does:**
- **Goal interpretation** – understands what user actually wants
- **Planning** – breaks goals into sub-goals
- **Reasoning** – during both planning and execution
- **Tool selection** – decides which tool to use for which task
- **Communication** – generates natural language for human interaction

```python
# Simple Brain implementation using an LLM
class AgentBrain:
    def __init__(self, llm):
        self.llm = llm  # e.g., GPT-4, Claude, Gemini
    
    def interpret_goal(self, user_input):
        prompt = f"Interpret this goal and extract key requirements: {user_input}"
        return self.llm.generate(prompt)
        # Example output: {"action": "hire", "role": "backend engineer", 
        #                  "experience": "2-4 years", "remote": True}
    
    def plan(self, goal):
        prompt = f"Break this goal into a sequence of steps: {goal}"
        return self.llm.generate(prompt).split("\n")
        # Example output: ["draft_JD", "post_on_LinkedIn", "shortlist", "interview", "send_offer"]
    
    def reason(self, observation, context):
        prompt = f"Given {observation} and context {context}, what should I do next?"
        return self.llm.generate(prompt)
    
    def select_tool(self, task):
        prompt = f"Which tool is best for: {task}? Available tools: search, email, calendar, resume_parser"
        return self.llm.generate(prompt).strip()  # Returns tool name
    
    def communicate(self, message, user_type="human"):
        # Generates human-friendly response
        return self.llm.generate(f"Respond politely to user: {message}")
```

---

## 2. Orchestrator

**Definition:** The "project manager" or "nervous system" – executes the plan step by step, handling sequencing, routing, retries, loops, and delegation.

**What the Orchestrator does:**
- **Task sequencing** – decides order of execution
- **Conditional routing** – based on previous step output, chooses next step
- **Retry logic** – if a tool fails, retries or finds alternative
- **Looping/iteration** – repeats steps when needed
- **Delegation** – decides when to ask human vs LLM vs tool

```python
class Orchestrator:
    def __init__(self, brain, tools, memory, supervisor):
        self.brain = brain
        self.tools = tools
        self.memory = memory
        self.supervisor = supervisor
    
    def execute_plan(self, plan):
        """Executes plan with conditional routing, retries, loops"""
        step_index = 0
        while step_index < len(plan):
            step = plan[step_index]
            
            # Conditional routing based on previous result
            if step == "post_on_LinkedIn" and self.memory.get("jd_drafted") == False:
                step_index += 1  # Skip if JD not ready
                continue
            
            # Retry logic
            for attempt in range(3):
                try:
                    result = self.execute_step(step)
                    break  # Success, exit retry loop
                except ToolFailureError:
                    if attempt == 2:
                        # After 3 failures, ask human
                        result = self.supervisor.escalate(f"Step {step} failed after 3 attempts")
            
            # Store result in memory
            self.memory.store(f"step_{step}_result", result)
            
            # Conditional routing based on result
            if step == "shortlist" and result["strong_candidates"] == 0:
                step_index = plan.index("broaden_JD")  # Jump to adaptation step
                continue
            
            # Looping - repeat interview step for each candidate
            if step == "interview" and result["more_candidates"]:
                # Don't increment step_index – repeat this step
                continue
            
            step_index += 1
    
    def execute_step(self, step):
        # Delegate to appropriate executor
        if step in ["draft_JD", "plan"]:
            return self.brain.plan(step)  # Brain handles cognitive steps
        elif step in ["post_job", "send_email", "check_calendar"]:
            return self.tools.use(step)   # Tools handle actions
        elif step in ["approve_offer"]:
            return self.supervisor.request_approval(step)  # Supervisor handles human
        else:
            return self.default_execute(step)
```

---

## 3. Tools (Hands and Legs)

**Definition:** Components that allow the agent to interact with the external world – APIs, databases, search engines, email, calendar, RAG knowledge bases.

```python
class ToolBox:
    def __init__(self):
        self.tools = {
            "linkedin_api": LinkedInAPI(),
            "resume_parser": ResumeParser(),
            "calendar_api": CalendarAPI(),
            "email_api": EmailAPI(),
            "search": SearchTool(),
            "rag_knowledge_base": RAGKnowledgeBase()  # Company documents
        }
    
    def use(self, tool_name, params):
        if tool_name not in self.tools:
            raise ToolNotFoundError(f"{tool_name} not available")
        return self.tools[tool_name].execute(params)

# Example tool implementations
class LinkedInAPI:
    def execute(self, params):
        if params["action"] == "post_job":
            return self.post_job(params["jd"])
        elif params["action"] == "boost_post":
            return self.boost_ad(params["job_id"], params["budget"])
        elif params["action"] == "get_applications":
            return self.fetch_applications(params["job_id"])
        return {"status": "success"}

class ResumeParser:
    def execute(self, params):
        resume_text = self.extract_text(params["resume_pdf"])
        return {
            "skills": ["Python", "Django", "AWS"],
            "experience_years": 3.5,
            "education": "B.Tech CS",
            "match_score": 0.85
        }

class CalendarAPI:
    def execute(self, params):
        if params["action"] == "get_free_slots":
            return ["2024-01-15 10:00", "2024-01-15 14:00"]
        elif params["action"] == "schedule":
            return {"meeting_id": "abc123", "status": "scheduled"}

class RAGKnowledgeBase:
    """Retrieves company-specific information"""
    def __init__(self, vector_db, company_docs):
        self.vector_db = vector_db
        self.docs = company_docs  # Past JDs, salary bands, policies
    
    def execute(self, params):
        query = params["query"]
        relevant_docs = self.vector_db.search(query, top_k=3)
        return {"retrieved_docs": relevant_docs}

# Using tools in agent
tools = ToolBox()
result = tools.use("linkedin_api", {"action": "post_job", "jd": jd_text})
result = tools.use("resume_parser", {"resume_pdf": candidate_resume})
result = tools.use("calendar_api", {"action": "get_free_slots"})
```

---

## 4. Memory

**Definition:** Stores information across the agent's lifespan – both short-term (current session) and long-term (persistent across sessions).

**Two types of memory:**

| Type | Stores | Persistence | Example |
|------|--------|-------------|---------|
| **Short-term** | Current session messages, tool call results, immediate decisions | Session only | "User asked for JD at 10:00 AM", "LinkedIn post ID = 12345" |
| **Long-term** | High-level goals, past interactions, user preferences, policies, guardrails | Across sessions | "Company prefers remote candidates", "Budget = $5000" |

```python
class AgentMemory:
    def __init__(self):
        self.short_term = {
            "session_id": "abc-123",
            "messages": [],           # Conversation history
            "tool_responses": {},     # Results from tool calls
            "current_step": "draft_JD",
            "progress": {
                "jd_drafted": True,
                "job_posted": False,
                "applications": 0
            }
        }
        
        self.long_term = {
            "user_preferences": {
                "remote_only": True,
                "preferred_platforms": ["LinkedIn", "Naukri"],
                "max_budget": 5000
            },
            "company_policies": {
                "salary_band_2_4_years": "8-12 LPA",
                "no_weekend_interviews": True,
                "required_tech_stack": ["Python", "Django", "AWS"]
            },
            "past_hiring_data": {
                "successful_sources": ["LinkedIn", "Referrals"],
                "avg_time_to_hire": 14  # days
            },
            "guardrails": [
                "never_send_offer_without_approval",
                "never_exceed_budget"
            ]
        }
    
    def store(self, key, value, persistent=False):
        if persistent:
            self.long_term[key] = value
        else:
            self.short_term[key] = value
    
    def recall(self, key):
        # Check short-term first, then long-term
        if key in self.short_term:
            return self.short_term[key]
        return self.long_term.get(key, None)
    
    def update_progress(self, task, status):
        self.short_term["progress"][task] = status
    
    def get_context_prompt(self):
        """Generate context for LLM from memory"""
        return f"""
        Current progress: {self.short_term['progress']}
        User preferences: {self.long_term['user_preferences']}
        Guardrails: {self.long_term['guardrails']}
        Previous conversation: {self.short_term['messages'][-5:]}
        """
```

---

## 5. Supervisor (Human-in-the-Loop)

**Definition:** Component that enables collaboration between the agent and humans – handles approvals, guardrails, and escalations.

```python
class Supervisor:
    def __init__(self, notification_channel="email", human_contact="hr@company.com"):
        self.notification_channel = notification_channel
        self.human_contact = human_contact
    
    def request_approval(self, action, details):
        """Ask human before executing high-risk actions"""
        print(f"Approval required for: {action}")
        print(f"Details: {details}")
        
        if self.notification_channel == "email":
            self.send_approval_email(action, details)
        
        user_input = input(f"Approve {action}? (y/n/explain): ")
        if user_input.lower() == 'y':
            return {"status": "approved", "action": action}
        elif user_input.lower() == 'n':
            return {"status": "rejected", "action": action}
        else:
            # Human gave explanation – agent should adapt
            return {"status": "needs_modification", "feedback": user_input}
    
    def enforce_guardrail(self, proposed_action, context):
        """Check if action violates any guardrail"""
        guardrails = context["long_term_memory"].get("guardrails", [])
        
        for rule in guardrails:
            if rule == "never_send_offer_without_approval" and proposed_action == "send_offer":
                return self.request_approval("send_offer", context)
            
            if rule == "never_exceed_budget" and proposed_action == "boost_ads":
                current_spend = context["short_term_memory"].get("ad_spend", 0)
                if current_spend + context["proposed_budget"] > context["budget_limit"]:
                    return self.request_approval("increase_budget", {"current": current_spend, "proposed": context["proposed_budget"]})
        
        return {"status": "allowed", "action": proposed_action}
    
    def escalate(self, issue, context):
        """For edge cases – alert human with recommendation"""
        message = f"""
        ALERT: Agent encountered an issue that requires human attention.
        
        Issue: {issue}
        Current context: {context}
        
        Suggested action: Please review manually.
        """
        self.send_notification(message)
        
        # Wait for human response (in real system, this would be async)
        return self.wait_for_human_response()
    
    def handle_edge_case(self, candidate, context):
        """Example: Candidate from non-IIT/NIT but excellent profile"""
        if candidate["college"] not in ["IIT", "NIT"] and candidate["match_score"] > 0.9:
            return self.escalate(
                "Strong candidate from non-premier institute",
                {"candidate": candidate, "recommendation": "Consider bypassing institute filter"}
            )
        return {"decision": "auto_process"}
```

---

## Complete Agentic AI System – All Components Together

```python
class CompleteAgenticAISystem:
    def __init__(self, llm_model, tools, memory, supervisor):
        self.brain = AgentBrain(llm_model)
        self.orchestrator = Orchestrator(self.brain, tools, memory, supervisor)
        self.tools = tools
        self.memory = memory
        self.supervisor = supervisor
    
    def run(self, user_goal):
        # 1. Brain interprets goal
        interpreted_goal = self.brain.interpret_goal(user_goal)
        self.memory.store("goal", interpreted_goal, persistent=True)
        
        # 2. Brain creates plan (multiple candidate plans, then selects best)
        candidate_plans = self.brain.generate_candidate_plans(interpreted_goal)
        best_plan = self.evaluate_and_select_plan(candidate_plans)
        self.memory.store("current_plan", best_plan)
        
        # 3. Orchestrator executes plan step by step
        final_result = self.orchestrator.execute_plan(best_plan)
        
        # 4. Supervisor handles any approvals needed during execution
        #    (already embedded in orchestrator)
        
        return final_result

# Usage example
system = CompleteAgenticAISystem(
    llm_model=GPT4(),
    tools=ToolBox(),
    memory=AgentMemory(),
    supervisor=Supervisor()
)

result = system.run("Hire a remote backend engineer with 2-4 years experience")
print(result)  # "Successfully hired candidate. Onboarding triggered."
```

---

## Summary Table: 5 Components

| Component | Role | Example |
|-----------|------|---------|
| **Brain (LLM)** | Intelligence – plans, reasons, selects tools | GPT-4, Claude, Gemini |
| **Orchestrator** | Execution manager – sequences, routes, retries | LangGraph, CrewAI, AutoGen |
| **Tools** | External actions – APIs, search, RAG, email | LinkedIn API, Resume parser, Calendar |
| **Memory** | Storage – short-term & long-term state | Conversation history, user preferences |
| **Supervisor** | Human collaboration – approvals, guardrails, escalations | Approval requests, budget checks |

---

## Key Takeaway

> **Agentic AI = Brain (LLM) + Orchestrator (Framework) + Tools (APIs) + Memory (State) + Supervisor (Human-in-Loop)**

> All five components work together to create a system that is **autonomous, goal-oriented, planning, reasoning, adaptable, and context-aware**.

### Key Characteristics of AI Agent :-
- Autonomus
- Goal Oriented
- Planning
- Reasoning
- Adaptability
- Context Awarness

### Key Components of AI Agent :-
- Brain
- Orchestrator
- Tools
- Memory
- Supervisor

---

## 04. LangChain Vs LangGraph (01:27:28)

This lecture explains **why LangGraph was created** by comparing it with LangChain. It uses an **automated hiring workflow** (from the previous video) to show that LangChain works well for **linear workflows** but struggles with **non‑linear workflows** (loops, conditionals, jumps). LangGraph solves this by representing every workflow as a **graph** – nodes (tasks) and edges (control flow) – eliminating the need for messy “glue code”.

---

## 📌 Important Pointers

| # | Key Point |
|---|------------|
| 1 | **LangChain** = open‑source library for building LLM‑based apps. Great for **linear chains** (step A → B → C). |
| 2 | **LangChain components**: Models (unified LLM interface), Prompts, Retrievers (RAG), Chains (sequence of components). |
| 3 | **Automated hiring workflow** (from earlier video) is **non‑linear**: has conditional branches, loops, and jumps. |
| 4 | Implementing this in **LangChain** requires writing custom Python `while` loops, `if‑else`, etc. – this extra code is called **glue code**. |
| 5 | Glue code makes the codebase **hard to maintain, debug, and scale** – especially in teams. |
| 6 | **LangGraph** represents every workflow as a **graph**: nodes = tasks (Python functions), edges = control flow. |
| 7 | LangGraph provides **built‑in constructs** for conditional edges, loops, and jumps – **no glue code** needed. |
| 8 | LangGraph is built by the LangChain team and is one of the top frameworks for **agentic AI** (along with CrewAI, AutoGen). |

---

## 1. Quick Recap: What is LangChain?

**Definition:** LangChain is an open‑source library that simplifies building applications powered by LLMs (Large Language Models).

**Core building blocks (components):**

| Component | Purpose |
|-----------|---------|
| **Models** | Unified interface to talk to any LLM (OpenAI, Anthropic, Hugging Face, Ollama, etc.) |
| **Prompts** | Help with prompt engineering (templates, few‑shot examples) |
| **Retrievers** | Fetch relevant documents from vector stores / knowledge bases (for RAG) |
| **Chains** | Combine components in sequence – output of one becomes input of the next. **This is LangChain’s flagship feature.** |

**What you can build with LangChain:**
- Simple conversational chatbots
- Text summarizers
- Multi‑step workflows (e.g., generate a detailed report, then summarise it)
- RAG applications (chat with your documents)
- **Basic agents** (LLM + tools, e.g., weather API)

**Simple linear chain example (LangChain):**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Chain 1: Generate a report
report_prompt = ChatPromptTemplate.from_template(
    "Write a detailed report about: {topic}"
)
report_chain = report_prompt | llm | StrOutputParser()

# Chain 2: Summarise the report
summary_prompt = ChatPromptTemplate.from_template(
    "Summarise this report in 3 bullet points:\n{report}"
)
summary_chain = summary_prompt | llm | StrOutputParser()

# Combined chain (linear)
full_chain = report_chain | summary_chain

result = full_chain.invoke({"topic": "remote work trends"})
print(result)
```

**Note:** This is linear – A → B → C. No loops, no conditionals.

---

## 2. The Automated Hiring Workflow (Non‑Linear)

This is the workflow from the previous video (simplified here):

```mermaid
flowchart TD
    START([START]) --> A[Hiring Request]
    A --> B[Create JD]
    B --> C{JD Approved?}
    C -->|No| B
    C -->|Yes| D[Post JD on LinkedIn/Naukri]
    D --> E[Wait 7 days]
    E --> F{Applications >= 20?}
    F -->|No| G[Modify JD]
    G --> H[Wait 48 hours]
    H --> F
    F -->|Yes| I[Shortlist candidates]
    I --> J[Schedule interviews]
    J --> K[Conduct interviews]
    K --> L{Selected?}
    L -->|Yes| M[Send offer letter]
    L -->|No| N[Send rejection email]
    M --> O{Accepted?}
    O -->|No| P[Renegotiate]
    P --> M
    O -->|Yes| Q[Onboarding]
    Q --> END([END])
    N --> END
```

**Why this is non‑linear:**
- **Conditional branches** – “JD Approved?” (yes/no), “Applications >= 20?” etc.
- **Loops** – “JD not approved → go back to Create JD”, “Low applications → modify JD → wait → re‑check”
- **Jumps** – after modifying JD, jump back to the monitoring step.

---

## 3. Trying to Implement This in LangChain – The Problem

LangChain’s native construct is the **linear chain**. To implement loops or conditionals, you have to write **plain Python** around the chains. That extra code is called **glue code**.

**Example: implementing only the JD approval loop in LangChain**

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Helper functions (glue code starts here)
def create_jd(prompt_text):
    jd_prompt = ChatPromptTemplate.from_template(
        "Create a job description for: {request}"
    )
    chain = jd_prompt | llm | StrOutputParser()
    return chain.invoke({"request": prompt_text})

def approve_jd(jd_text):
    # Dummy approval logic – in reality, ask a human
    return "engineer" in jd_text.lower()

def post_jd(jd_text):
    print("Posting JD to LinkedIn...")
    # API call would go here

# Main workflow with loop (glue code continues)
hiring_prompt = "Need a backend engineer, remote, 2-4 years exp."

approved = False
jd = None
while not approved:
    jd = create_jd(hiring_prompt)
    print("\nGenerated JD:\n", jd)
    approved = approve_jd(jd)
    if not approved:
        print("JD not approved. Regenerating...\n")

post_jd(jd)
```

**Problems with this approach:**
- The `while` loop and `if` conditions are **hand‑written Python**, not LangChain abstractions.
- As the workflow grows (multiple loops, branches, nested conditions), the glue code becomes a **spaghetti** of custom logic.
- Hard to **debug** (errors can be anywhere in the Python code).
- Hard to **maintain** (changing one step may break manual loop indexes).
- **Not scalable** for complex agentic workflows.

> **Glue code = any code you write to stitch together library components that the library itself doesn’t provide. Less glue code = better maintainability.**

---

## 4. LangGraph – Workflow as a Graph

LangGraph (by the LangChain team) solves this by letting you represent your workflow as a **graph**:

- **Nodes** = individual tasks (Python functions)
- **Edges** = control flow between nodes (including conditional edges, loops, jumps)

**Key advantages:**
- No manual loops / if‑else – you declare the graph structure once.
- Built‑in support for **conditional routing**, **looping back**, **parallel execution**.
- **Zero glue code** – everything is expressed in LangGraph’s graph primitives.
- Much easier to **visualise, debug, and modify**.

### Basic LangGraph Example (Same JD Approval Loop)

```python
from typing import Literal, TypedDict
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Define the state that flows through the graph
class HiringState(TypedDict):
    request: str
    jd: str
    approved: bool

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Node 1: Create JD
def create_jd(state: HiringState) -> HiringState:
    prompt = ChatPromptTemplate.from_template(
        "Create a job description for: {request}"
    )
    chain = prompt | llm | StrOutputParser()
    jd = chain.invoke({"request": state["request"]})
    print("\n--- Generated JD ---\n", jd)
    return {**state, "jd": jd}

# Node 2: Check approval (dummy logic)
def check_approval(state: HiringState) -> HiringState:
    approved = "engineer" in state["jd"].lower()
    print(f"Approved: {approved}")
    return {**state, "approved": approved}

# Node 3: Post JD
def post_jd(state: HiringState) -> HiringState:
    print("\n✅ Posting approved JD to job portals...")
    # API call would go here
    return state

# Conditional router
def approval_router(state: HiringState) -> Literal["approved", "not_approved"]:
    return "approved" if state["approved"] else "not_approved"

# Build the graph
graph = StateGraph(HiringState)
graph.add_node("create_jd", create_jd)
graph.add_node("check_approval", check_approval)
graph.add_node("post_jd", post_jd)

# Edges
graph.set_entry_point("create_jd")
graph.add_edge("create_jd", "check_approval")
graph.add_conditional_edges(
    "check_approval",
    approval_router,
    {
        "approved": "post_jd",       # if approved → go to post_jd
        "not_approved": "create_jd"  # if not approved → loop back to create_jd
    }
)
graph.add_edge("post_jd", END)

# Compile and run
app = graph.compile()
initial_state = {"request": "Need a backend engineer, remote, 2-4 years exp.", "jd": "", "approved": False}
final_state = app.invoke(initial_state)
```

**What makes this better:**
- The **loop** is expressed as a **conditional edge** from `check_approval` back to `create_jd`.
- No `while` loop, no manual `if` for routing – the graph runtime handles it.
- All control flow is **visible** in the graph definition.
- Adding more branches (e.g., different handling for “low applications”) is just more nodes and edges.

---

## 5. Summary: LangChain vs LangGraph

| Aspect | LangChain | LangGraph |
|--------|-----------|-----------|
| **Primary use** | Linear workflows (chains) | Non‑linear workflows (graphs) |
| **Control flow** | Only sequential (A→B→C) | Conditional branches, loops, jumps, parallel execution |
| **Implementation of loops/conditionals** | Requires custom Python glue code (`while`, `if-else`) | Built‑in via conditional edges and graph structure |
| **Glue code** | High for complex workflows | Zero (all logic expressed in graph) |
| **Maintainability** | Low for complex workflows | High – graph is declarative and visualisable |
| **Best for** | Simple RAG, chatbots, summarisation chains | Agentic AI, multi‑step decision workflows, human‑in‑the‑loop |

---

## 6. Why This Matters for Agentic AI

Agentic AI systems (like the automated hiring agent) are inherently **non‑linear**:
- They **plan** dynamically (which depends on the goal).
- They **react** to tool outputs and environment feedback.
- They **adapt** by changing their plan at runtime.

Trying to build such systems with LangChain alone leads to an unmaintainable mess of glue code. **LangGraph provides the missing graph‑based orchestration layer** – the “orchestrator” component we discussed in the previous video.

> **Quote from the video:**  
> *“LangChain works really well with linear workflows (chains). But as soon as non‑linearity enters the system, LangChain gives up. LangGraph was built to solve exactly that.”*

---

## Workflow Challenges: Control flow, State Handling, Event-Driven Execution, Fault Tolerance, and Human-in-the-Loop – Why LangGraph Excels

This lecture explains **why LangGraph is superior to LangChain for building complex, long‑running, non‑linear workflows** using the automated hiring example. The instructor walks through **seven major challenges** that you face when using LangChain, and shows how LangGraph solves each one **out‑of‑the‑box** with **zero glue code**.

---

## 📌 Important Pointers (TL;DR)

| # | Challenge | LangChain Problem | LangGraph Solution |
|---|-----------|-------------------|--------------------|
| 1 | **Control flow complexity** | Only linear chains – no built‑in loops, conditionals, or jumps | Graph nodes + conditional edges + natural loops |
| 2 | **State management** | Stateless – no native key‑value store across steps | Stateful – shared state dictionary accessible by all nodes |
| 3 | **Event‑driven execution** | Cannot pause for hours/days; requires splitting chains and manual scheduling | Built‑in `interrupt_after` + checkpointing; resumes after external trigger |
| 4 | **Fault tolerance** | No retry or recovery – crash means restart from beginning | Automatic retry (node‑level) + recovery from last checkpoint (system‑level) |
| 5 | **Human‑in‑the‑loop** | Only synchronous `input()` – blocks process for long periods | First‑class support – pause indefinitely, resume when human input arrives |
| 6 | **Nested workflows** | Not possible | Subgraphs – a graph can be used as a node in another graph (multi‑agent, reusability) |
| 7 | **Observability** | Only traces LangChain parts, not custom glue code | Full tracing of every node, state change, and human interaction via LangSmith |

---

## 1. Control Flow Complexity

### What is the problem?
LangChain is designed for **linear chains** (A → B → C). Complex workflows have:
- **Conditional branches** (if‑else)
- **Loops** (repeat until condition)
- **Jumps** (go back or forward to any node)

### LangChain approach (bad – glue code required)
```python
# You have to write plain Python loops and conditionals
approved = False
while not approved:
    jd = create_jd_chain.invoke(...)   # LangChain part
    approved = approve_jd(jd)           # custom function
    if not approved:
        print("Regenerating...")
post_jd(jd)                             # custom function
```
**Problem:** This is **glue code** – not part of LangChain. As you add more branches and loops, the code becomes a spaghetti of custom logic, hard to maintain and debug.

### LangGraph solution – declarative graph
```python
from langgraph.graph import StateGraph, END

graph = StateGraph(State)
graph.add_node("create_jd", create_jd_fn)
graph.add_node("check_approval", check_approval_fn)
graph.add_node("post_jd", post_jd_fn)

graph.set_entry_point("create_jd")
graph.add_edge("create_jd", "check_approval")
graph.add_conditional_edges(
    "check_approval",
    lambda state: "approved" if state["approved"] else "not_approved",
    {
        "approved": "post_jd",
        "not_approved": "create_jd"   # loop back!
    }
)
graph.add_edge("post_jd", END)
```
✅ No manual loops, no glue code – the graph runtime handles everything.

---

## 2. State Management

### What is “state”?
State = all the data that your workflow needs to remember and update over time. In the hiring workflow, state includes:
- `jd` (job description text)
- `jd_approved` (boolean)
- `application_count` (integer)
- `shortlisted_candidates` (list)
- `offer_status` (string)

### LangChain problem – stateless
LangChain has **conversational memory** (stores chat history as text) but no way to store arbitrary key‑value pairs across steps. You have to manually create a Python dictionary and pass it around.

```python
# Manual state handling in LangChain (glue code)
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm = ChatOpenAI(model="gpt-4", temperature=0)

# Manual state dictionary
state = {
    "jd": None,
    "approved": False,
    "posted": False
}

# Step 1: Create JD
jd_prompt = ChatPromptTemplate.from_template("Create JD for: {request}")
jd_chain = jd_prompt | llm | StrOutputParser()
state["jd"] = jd_chain.invoke({"request": "Hire backend engineer"})

# Step 2: Approval (manual update)
user_input = input("Approve JD? (y/n): ")
state["approved"] = (user_input == "y")

# Step 3: Post JD (manual update)
if state["approved"]:
    print("Posting JD...")
    state["posted"] = True

print("Final state:", state)
```
❌ Error‑prone, verbose, and doesn’t scale.

### LangGraph solution – stateful graph
You define a state schema (TypedDict or Pydantic), and every node receives the current state and returns an updated state. The graph automatically propagates it.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

# Define state schema
class HiringState(TypedDict):
    request: str
    jd: str
    approved: bool
    posted: bool

# Node functions – receive state, return updated state
def create_jd(state: HiringState) -> HiringState:
    # Simulate JD generation
    jd_text = f"JD for {state['request']}..."
    return {**state, "jd": jd_text}

def approve_jd(state: HiringState) -> HiringState:
    # In real code, this could ask a human or check rules
    approved = "engineer" in state["jd"].lower()
    return {**state, "approved": approved}

def post_jd(state: HiringState) -> HiringState:
    print(f"Posting JD: {state['jd']}")
    return {**state, "posted": True}

# Build graph
graph = StateGraph(HiringState)
graph.add_node("create_jd", create_jd)
graph.add_node("approve_jd", approve_jd)
graph.add_node("post_jd", post_jd)

graph.set_entry_point("create_jd")
graph.add_edge("create_jd", "approve_jd")
graph.add_edge("approve_jd", "post_jd")
graph.add_edge("post_jd", END)

# Compile and run
app = graph.compile()
initial_state = {"request": "Hire backend engineer", "jd": "", "approved": False, "posted": False}
final_state = app.invoke(initial_state)
print(final_state)
```
✅ State is **shared, mutable, and automatically managed** – no manual dictionary passing.

---

## 3. Event‑Driven Execution

### What is event‑driven execution?
Instead of running from start to finish without stopping, the workflow **pauses** and waits for an **external trigger** (time passing, human action, API callback). Examples in hiring:
- Wait **7 days** after posting the JD.
- Wait for candidate to **accept/reject** the offer.

### LangChain problem – no pause/resume
LangChain chains run **synchronously and linearly**. To wait for 7 days, you must:
- Split the chain into two separate chains.
- Use an external scheduler (cron job) to run the second part.
- Manually save and restore state between runs (again, glue code).

### LangGraph solution – checkpointing + interrupt
```python
from langgraph.checkpoint import MemorySaver

# Create graph with checkpointing
checkpointer = MemorySaver()
graph = StateGraph(HiringState)
# ... add nodes and edges ...
app = graph.compile(checkpointer=checkpointer)

# Run until interruption
config = {"configurable": {"thread_id": "hiring-1"}}
for event in app.stream(initial_state, config, interrupt_after=["post_jd"]):
    print(event)  # Execution stops after 'post_jd' node

# Later, after 7 days, resume from the same checkpoint
for event in app.stream(None, config, resume=True):
    print(event)  # Continues from where it stopped
```
✅ Built‑in pause/resume – the state is saved and restored automatically.

---

## 4. Fault Tolerance

### Two types of failures:
- **Small (node‑level)** – an API call fails temporarily (e.g., LinkedIn API down).
- **Large (system‑level)** – the whole server crashes or the container restarts.

### LangChain problem – no fault tolerance
If a chain fails at step 3, you **lose all progress** and must start over from step 1. No built‑in retry or recovery.

### LangGraph solution – retry + recovery via checkpoints

**Retry (small failures):**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def post_jd_to_linkedin(state):
    # This will retry up to 3 times if the API fails
    return linkedin_api.post(state["jd"])
```

**Recovery (system crash):**
Because every step is checkpointed, you can resume from the **last completed node** after a crash.

```python
# After a crash, restart the app with the same thread ID
app = graph.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "hiring-1"}}
final_state = app.invoke(None, config)   # resumes from last checkpoint
```
✅ No progress lost – like a “save game” for your workflow.

---

## 5. Human‑in‑the‑Loop

### What is it?
The workflow pauses and asks a human for input (approval, decision) before continuing. Needed for risky actions (sending offer letters, posting jobs).

### LangChain problem – only synchronous, short input
You can use `input()`, but that **blocks** the process. If the human takes 24 hours, the script keeps running, wasting resources and risking crashes.

### LangGraph solution – asynchronous, indefinite pause
```python
from langgraph.checkpoint import MemorySaver
from langgraph.graph import StateGraph, END

def human_approval_node(state: HiringState) -> HiringState:
    # This node will be interrupted for human input
    # In practice, you'd send an email/slack notification
    print("Waiting for human approval...")
    # The graph will stop here and save state
    return state

# Build graph with interrupt before the approval node
graph = StateGraph(HiringState)
graph.add_node("create_jd", create_jd)
graph.add_node("human_approval", human_approval_node)
graph.add_node("post_jd", post_jd)

graph.set_entry_point("create_jd")
graph.add_edge("create_jd", "human_approval")
graph.add_conditional_edges(
    "human_approval",
    lambda state: "approved" if state.get("human_approved") else "pending",
    {"approved": "post_jd", "pending": END}
)

# Compile with checkpointer
app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "hiring-1"}}

# Run until human input is needed
for event in app.stream(initial_state, config, interrupt_after=["human_approval"]):
    print(event)

# Later, when human provides input via an external API
human_decision = "approved"  # from webhook or CLI
# Update the state with human decision and resume
app.update_state(config, {"human_approved": True})
for event in app.stream(None, config, resume=True):
    print(event)  # Continues to post_jd
```

### Key Features :-
- Pause **indefinitely** (minutes, hours, days), resume exactly where you left off.
- The state is persisted across server restarts.
- Human can provide input asynchronously (via email, Slack, web dashboard).
- After input, the graph resumes exactly where it left off.

---

## 6. Nested Workflows (Subgraphs)

### What is it?
A node in a graph can itself be another graph. This enables:
- **Multi‑agent systems** – different agents collaborate (e.g., sensor agent, driving agent, entertainment agent in a self‑driving car).
- **Reusability** – create a reusable “approval workflow” and use it in multiple places.

### LangChain problem – impossible
LangChain has no concept of graphs, let alone nested ones.

### LangGraph solution – subgraphs
```python
# Define a reusable approval subgraph
approval_graph = StateGraph(ApprovalState)
# ... add nodes for approval logic ...

# Main graph uses the subgraph as a node
main_graph = StateGraph(MainState)
main_graph.add_node("approve_jd", approval_graph.compile())  # subgraph as node
main_graph.add_node("approve_offer", approval_graph.compile()) # reused
```
✅ You can build complex systems by composing smaller, reusable graphs.

---

## 7. Observability (LangSmith Integration)

### Why is observability important?
When your agent acts unexpectedly (e.g., spends too much money on ads), you need to **audit** what happened – which decisions were made, what state changes occurred, what inputs were given.

### LangChain + LangSmith – partial observability
LangSmith can trace LangChain chain executions (LLM calls, prompts, responses), but it **cannot trace your custom glue code** (loops, conditionals, manual state updates). So you get only **partial** visibility.

### LangGraph + LangSmith – full observability
Because LangGraph has **no glue code** – everything is expressed as nodes, edges, and state – LangSmith can track:
- Every node execution (order, timing)
- State before and after each node
- Human inputs and approvals
- Full timeline of events

```python
# Just add a callback to LangSmith
from langsmith import Client
client = Client()
# LangGraph automatically sends all events to LangSmith
```
✅ Complete **tracing and debugging** – essential for production agentic systems.

---

## Final Comparison Table

| Feature | LangChain | LangGraph |
|---------|-----------|-----------|
| **Control flow** | Linear chains only | Graphs with loops, conditionals, jumps |
| **State management** | Manual (glue code) | Built‑in stateful propagation |
| **Event‑driven execution** | Not possible (requires splitting) | `interrupt_after/ before` + checkpointing |
| **Fault tolerance** | None (restart from beginning) | Retry + recovery from checkpoint |
| **Human‑in‑the‑loop** | Only synchronous `input()` | Asynchronous, indefinite pause |
| **Nested workflows** | Not possible | Subgraphs (multi‑agent, reusability) |
| **Observability** | Partial (only LangChain parts) | Full (every node, state, human interaction) |
| **Glue code** | High (for complex workflows) | Zero |

---

## Key Lesson

> *“LangChain works really well with linear workflows (chains). But as soon as non‑linearity enters the system, LangChain gives up. LangGraph was built to solve exactly that – it makes your workflow a first‑class graph, with state, checkpoints, and human‑in‑the‑loop built right in.”*

---

## Conclusion & Revision – LangGraph vs LangChain

This final section of the tutorial **recaps everything** learned about LangGraph, clarifies **when to use LangChain vs LangGraph**, and emphasizes that **LangGraph is not a replacement** for LangChain – it's an **orchestration layer** built on top of it.

---

## 📌 Important Pointers

| # | Key Point |
|---|-----------|
| 1 | **LangGraph** = orchestration framework for building **stateful, multi‑step, event‑driven** workflows with LLMs. |
| 2 | Think of LangGraph as a **flowchart engine for LLMs** – you define steps as **nodes**, connections as **edges**, and the runtime handles state, branching, looping, pause/resume, and fault recovery. |
| 3 | **Use LangChain** for simple, **linear** workflows (prompt chains, summarizers, basic RAG). |
| 4 | **Use LangGraph** for complex, **non‑linear** workflows requiring conditionals, loops, human‑in‑the‑loop, multi‑agent coordination, or event‑driven execution. |
| 5 | **LangGraph is NOT a replacement for LangChain** – it is built **on top of LangChain**. You still need LangChain components: models, prompts, retrievers, document loaders, splitters, tools. |
| 6 | LangChain provides the **components**; LangGraph provides the **orchestration** to wire them together in complex ways. |
| 7 | Both work **hand‑in‑hand** – future videos will use **both** libraries together. |

---

## 1. What is LangGraph? (Final Definition)

> **LangGraph is an orchestration framework that enables you to build stateful, multi‑step, event‑driven workflows using LLMs. It's ideal for designing both single‑agent and multi‑agent agentic AI applications.**

**Think of LangGraph as a flowchart engine for LLMs.** You define:
- **Nodes** = individual steps (tasks)
- **Edges** = connections between steps (control flow)
- **Transition logic** = conditions, loops, jumps

LangGraph automatically handles:
- State management (sharing data across steps)
- Conditional branching (if‑else)
- Looping (repeat until condition)
- Pausing and resuming (wait for external triggers)
- Fault recovery (retry, checkpoint restart)

```python
# Basic LangGraph structure (conceptual)
from langgraph.graph import StateGraph, END

# 1. Define state
class MyState(TypedDict):
    data: str
    step_complete: bool

# 2. Define nodes (functions that update state)
def step_one(state: MyState) -> MyState:
    return {**state, "data": "processed"}

def step_two(state: MyState) -> MyState:
    return {**state, "step_complete": True}

# 3. Build graph
graph = StateGraph(MyState)
graph.add_node("step_one", step_one)
graph.add_node("step_two", step_two)
graph.set_entry_point("step_one")
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", END)

# 4. Compile and run
app = graph.compile()
result = app.invoke({"data": "", "step_complete": False})
```

---

## 2. When to Use What?

| Use **LangChain** when... | Use **LangGraph** when... |
|---------------------------|----------------------------|
| Simple **linear** workflows | Complex **non‑linear** workflows |
| Prompt chains | Need **conditional branches** (if‑else) |
| Text summarizers | Need **loops** (repeat until condition) |
| Basic RAG (retrieve → generate) | Need **human‑in‑the‑loop** (pause for approval) |
| No complex control flow | Need **multi‑agent coordination** (collaboration) |
| Short, synchronous tasks | Need **asynchronous, event‑driven** execution (wait hours/days) |

**Rule of thumb:**
- If your workflow can be drawn as a **straight line** (A→B→C→D), use **LangChain**.
- If your flowchart has **diamonds (decisions), arrows going backwards (loops), or pauses**, use **LangGraph**.

```python
# LangChain: linear chain
chain = prompt | llm | output_parser
result = chain.invoke({"topic": "AI"})

# LangGraph: non-linear with conditional loop
graph.add_conditional_edges(
    "check_approval",
    lambda s: "approved" if s["approved"] else "not_approved",
    {"approved": "next_step", "not_approved": "retry_step"}
)
```

---

## 3. Does LangGraph Replace LangChain?

**NO!** LangGraph is **built on top of LangChain**, not a replacement.

- **LangChain** provides the **building blocks**:
  - Chat models (`ChatOpenAI`, `ChatAnthropic`)
  - Prompt templates (`ChatPromptTemplate`)
  - Retrievers, document loaders, text splitters
  - Tools (API wrappers)
  - Output parsers

- **LangGraph** provides the **orchestration** to wire those blocks into complex, non‑linear workflows.

You will **always use both** in a typical agentic application:

```python
# Use LangChain components inside LangGraph nodes
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph

llm = ChatOpenAI(model="gpt-4")

def generate_jd(state):
    prompt = ChatPromptTemplate.from_template("Create JD for {role}")
    chain = prompt | llm
    jd = chain.invoke({"role": state["role"]})
    return {**state, "jd": jd.content}

# Then use LangGraph to orchestrate multiple such nodes
graph = StateGraph(State)
graph.add_node("generate_jd", generate_jd)
# ... add edges, conditionals, loops
```

**Key takeaway:** Your investment in learning LangChain is **not wasted** – you will still use it extensively inside LangGraph nodes.

---

## 4. Final Comparison Table

| Aspect | LangChain | LangGraph |
|--------|-----------|-----------|
| **Primary purpose** | Provide components (models, prompts, retrievers, tools) | Orchestrate complex workflows (graph execution) |
| **Workflow style** | Linear chains (A→B→C) | Non‑linear graphs (branches, loops, jumps) |
| **State management** | Stateless (manual dict) | Stateful (automatic propagation) |
| **Human‑in‑the‑loop** | Only synchronous `input()` | Asynchronous, indefinite pause/resume |
| **Event‑driven** | Not supported | Built‑in with checkpointing |
| **Fault tolerance** | None (restart from beginning) | Retry + recovery from checkpoints |
| **Multi‑agent** | Not possible | Yes (via subgraphs) |
| **Observability** | Partial (LangSmith traces chains only) | Full (traces every node, state change, human input) |
| **When to use** | Simple, linear, short‑running tasks | Complex, long‑running, interactive workflows |

---

## 5. Code Example: Both Libraries Working Together

This example shows how LangChain components are used **inside** a LangGraph node:

```python
# Import both
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict

# 1. LangChain components
llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = ChatPromptTemplate.from_template(
    "Write a short blog post about {topic}"
)
chain = prompt | llm | StrOutputParser()

# 2. LangGraph state
class BlogState(TypedDict):
    topic: str
    draft: str
    approved: bool

# 3. LangGraph node that uses LangChain chain
def write_draft(state: BlogState) -> BlogState:
    draft = chain.invoke({"topic": state["topic"]})
    return {**state, "draft": draft}

def human_approval(state: BlogState) -> BlogState:
    # In real app, this would pause and wait for external input
    print(f"Draft:\n{state['draft']}")
    approved = input("Approve? (y/n): ").lower() == 'y'
    return {**state, "approved": approved}

# 4. Build LangGraph
graph = StateGraph(BlogState)
graph.add_node("write", write_draft)
graph.add_node("approve", human_approval)
graph.set_entry_point("write")
graph.add_edge("write", "approve")
graph.add_conditional_edges(
    "approve",
    lambda s: "approved" if s["approved"] else "write",
    {"approved": END, "write": "write"}
)

app = graph.compile()
result = app.invoke({"topic": "AI agents", "draft": "", "approved": False})
print("Final approved draft:\n", result["draft"])
```

**Notice:** The chain (`prompt | llm | output_parser`) is pure LangChain. The loop and conditional routing are pure LangGraph. They work **together**.

---

## Key Lesson

> *“LangGraph is built on top of LangChain. LangChain gives you the components – models, prompts, retrievers, tools. LangGraph gives you the orchestration – how to wire them together in complex, non‑linear ways. **You will use both.** ”*

---

### Useful Links

- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)

- [Human-in-the-loop](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)

---

## 05. LangGraph Core Concepts (51:51)

This lecture provides a **quick revision of LangGraph** and then dives deep into the **core concept of LLM workflows** – what they are and the **five most common workflow patterns** you will encounter when building agentic applications.

---

## 📌 Important Pointers

| # | Key Point |
|---|------------|
| 1 | **LangGraph** is an **orchestration framework** that represents any LLM workflow as a **graph** – nodes = tasks, edges = control flow. |
| 2 | LangGraph features: parallel execution, loops, conditional branching, memory (state persistence), and resumability (fault tolerance). |
| 3 | **LLM Workflow** = a series of tasks executed in a specific order to achieve a goal, where many tasks involve calling LLMs. |
| 4 | **Prompt Chaining** – sequential LLM calls; output of one becomes input of the next. Used for complex tasks broken into steps. |
| 5 | **Routing** – an LLM classifies the input and sends it to a specialized handler (different LLM or logic). |
| 6 | **Parallelization** – split a task into independent subtasks, run them in parallel, then aggregate results. |
| 7 | **Orchestrator‑Workers** – similar to parallelization, but the subtasks are **dynamically determined** by an orchestrator LLM based on the input. |
| 8 | **Evaluator‑Optimizer** – iterative loop: a generator LLM creates a solution, an evaluator LLM gives feedback; repeat until acceptable. |
| 9 | LangGraph is **not a replacement for LangChain** – you will use **both** together. LangChain provides components (models, prompts, tools); LangGraph orchestrates them. |

---

## 1. What is LangGraph? (Quick Revision)

> **LangGraph is an orchestration framework for building intelligent, stateful, multi‑step LLM workflows.**

- It takes any LLM workflow and represents it as a **graph**.
- **Nodes** = individual tasks (e.g., call an LLM, call a tool, make a decision).
- **Edges** = control flow that determines which node runs next.

**Key features:**
- **Parallel execution** – multiple nodes can run simultaneously.
- **Loops** – nodes can go back to previous nodes (cycles).
- **Conditional branching** – choose next node based on a condition.
- **Memory** – persist state across steps (short‑term and long‑term).
- **Resumability** – if the workflow crashes, you can resume from the last checkpoint.

```python
# Conceptual LangGraph structure
from langgraph.graph import StateGraph, END

# Define state
class WorkflowState(TypedDict):
    data: str
    step_done: bool

# Define nodes (tasks)
def step_one(state: WorkflowState) -> WorkflowState:
    # do something
    return {**state, "step_done": True}

def step_two(state: WorkflowState) -> WorkflowState:
    # do something else
    return state

# Build graph
graph = StateGraph(WorkflowState)
graph.add_node("step_one", step_one)
graph.add_node("step_two", step_two)
graph.set_entry_point("step_one")
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", END)

# Compile and run
app = graph.compile()
result = app.invoke({"data": "input", "step_done": False})
```

---

## 2. What is an LLM Workflow?

**Workflow** = a series of tasks executed in a specific order to achieve a goal.  
**LLM Workflow** = a workflow where many of those tasks involve calling LLMs (e.g., generating text, reasoning, tool calling, decision making).

Example from previous video: **Automated Hiring Workflow**  
Tasks: draft JD → post JD → monitor applications → shortlist candidates → schedule interviews → conduct interviews → send offer → onboard. Each task may use an LLM.

Workflows can be:
- **Linear** – A → B → C
- **Parallel** – A splits into B and C running simultaneously
- **Branching** – if condition then D else E
- **Looped** – repeat a task until condition met

---

## 3. Five Common LLM Workflow Patterns

### 3.1 Prompt Chaining

**What it is:** Sequential LLM calls where the output of one becomes the input of the next.

**When to use:** When a complex task can be broken into a fixed sequence of simpler subtasks, and you want to check intermediate results.

**Example:** Generate a detailed report from a topic.
1. LLM #1: Create an outline from the topic.
2. (Optional) Validate outline length.
3. LLM #2: Generate a detailed report based on the outline.

```python
# Conceptual code for Prompt Chaining
def prompt_chain(topic):
    # Step 1: Generate outline
    outline = llm.invoke(f"Create an outline for a report on {topic}")
    
    # Optional validation
    if len(outline) > 1000:
        raise ValueError("Outline too long")
    
    # Step 2: Generate report from outline
    report = llm.invoke(f"Based on this outline, write a detailed report:\n{outline}")
    return report
```

---

### 3.2 Routing

**What it is:** An LLM (or a classifier) examines the input and **routes** it to a specialized handler (different LLM, different prompt, or different tool). Each handler is optimized for a specific type of request.

**When to use:** When you have different categories of requests (e.g., customer support: refund, technical, sales) and you want to use the best‑suited LLM for each.

**Example:** Customer support chatbot.
- Input query → Router LLM classifies as “refund”, “technical”, or “sales”.
- Then send to respective handler.

```python
# Conceptual Routing
def route_query(user_query):
    # Router LLM decides category
    category = router_llm.invoke(f"Classify this query into: refund, technical, sales.\nQuery: {user_query}")
    
    if category == "refund":
        return refund_handler(user_query)
    elif category == "technical":
        return technical_handler(user_query)
    else:
        return sales_handler(user_query)
```

---

### 3.3 Parallelization

**What it is:** Break a single task into **independent subtasks**, run them **in parallel** (simultaneously), then **aggregate** the results to produce a final output.

**When to use:** When multiple independent checks or evaluations must be performed on the same input, and they do not depend on each other.

**Example:** YouTube content moderation – check a video against three independent criteria:
- Community guidelines
- Misinformation
- Sexual content

All three checks can run at the same time. If any fails, the video is flagged.

```python
# Conceptual Parallelization (using asyncio or threads)
import asyncio

async def moderate_video(video_transcript):
    # Run three checks in parallel
    results = await asyncio.gather(
        check_guidelines(video_transcript),
        check_misinformation(video_transcript),
        check_sexual_content(video_transcript)
    )
    # Aggregate: if any fails, reject
    if all(results):
        return "APPROVED"
    else:
        return "FLAGGED"
```

---

### 3.4 Orchestrator‑Workers

**What it is:** Similar to parallelization, but the **subtasks are not fixed** – an **orchestrator LLM** dynamically decides what tasks to create and which worker LLM should execute them, based on the input.

**When to use:** When the nature of subtasks depends on the input, and you cannot pre‑define them.

**Example:** Research assistant that answers a complex query.
- Query: “Explain the impact of LLMs on software engineering.”
- Orchestrator decides: need to search Google Scholar for academic papers, search news for recent developments, maybe search GitHub for relevant repositories.
- Workers perform these parallel searches, then results are aggregated into a report.

```python
# Conceptual Orchestrator-Workers
def research_assistant(query):
    # Orchestrator decides what to search and where
    subtasks = orchestrator_llm.invoke(f"Given this query: '{query}', list the search tasks (e.g., 'search_google_scholar', 'search_news')")
    
    # Execute all subtasks in parallel
    results = []
    for task in subtasks:
        if task == "search_google_scholar":
            results.append(search_google_scholar(query))
        elif task == "search_news":
            results.append(search_news(query))
        # ... other tasks
    # Aggregate results into a final report
    return aggregate_results(results)
```

---

### 3.5 Evaluator‑Optimizer

**What it is:** An **iterative** workflow with two LLMs:
- **Generator** – creates an initial solution (e.g., email draft, blog post, poem).
- **Evaluator** – checks the solution against criteria, either **accepts** it or provides **feedback**.
- The generator then **refines** the solution based on feedback, and the loop repeats until the evaluator accepts.

**When to use:** For creative or open‑ended tasks where a perfect output is rarely achieved in one attempt (writing, code generation, design).

**Example:** Writing a professional email.
- Generator writes a draft.
- Evaluator checks for tone, length, clarity – if not satisfied, gives feedback (“too informal, add a greeting”).
- Generator rewrites.
- Loop continues until evaluator approves.

```python
# Conceptual Evaluator-Optimizer
def write_email(topic, max_iterations=5):
    solution = ""
    for i in range(max_iterations):
        if i == 0:
            solution = generator_llm.invoke(f"Draft an email about: {topic}")
        else:
            solution = generator_llm.invoke(f"Improve this email based on feedback:\nEmail: {solution}\nFeedback: {feedback}")
        
        # Evaluate
        evaluation, feedback = evaluator_llm.invoke(f"Evaluate this email (accept/reject). If reject, give feedback:\nEmail: {solution}")
        
        if evaluation == "accept":
            return solution
    return solution  # return last attempt even if not perfect
```

---

## 4. LangGraph vs LangChain – Final Clarification

| Aspect | LangChain | LangGraph |
|--------|-----------|-----------|
| **Purpose** | Provides components (models, prompts, retrievers, tools) | Orchestrates workflows (graphs) |
| **Workflow style** | Linear chains | Non‑linear graphs (loops, branches, parallel) |
| **State management** | Not built‑in (manual) | Built‑in stateful execution |
| **When to use** | Simple, linear tasks | Complex, non‑linear, long‑running workflows |

> **Important:** LangGraph is **built on top of LangChain**. You will **always use both** – LangChain components inside LangGraph nodes. Your LangChain knowledge is **not wasted**.

---

## Graphs, Nodes, Edges, State, Reducers, LangGraph Execution Model

This part of lecture covers the **fundamental concepts** of LangGraph: **graphs, nodes, edges, state, reducers, and the execution model**. The instructor uses a **UPSC essay evaluation workflow** as a practical example.

---

## 📌 Important Pointers

| # | Concept | Description |
|---|---------|-------------|
| 1 | **Graph** | The overall representation of an LLM workflow – a collection of nodes and edges. |
| 2 | **Node** | A single task in the workflow. Behind the scenes, each node is a **Python function**. |
| 3 | **Edge** | Defines the control flow – which node executes after which. Edges can be sequential, parallel, conditional, or looping. |
| 4 | **State** | Shared memory that flows through the workflow. Holds all key‑value data needed for execution (e.g., essay text, scores, feedback). State is **mutable** and **accessible by all nodes**. |
| 5 | **Reducer** | Determines **how updates to a state key are applied** – replace, add, merge, or custom logic. Essential for parallel tasks or when you need to preserve history (e.g., chat messages). |
| 6 | **Execution Model** | LangGraph’s runtime: **graph definition → compilation → invocation → message passing → supersteps**. Inspired by Google’s Pregel. |
| 7 | **Superstep** | A round of execution that may contain one or more **parallel steps**. When multiple nodes run simultaneously, they form a single superstep. |

---

## 1. Graphs, Nodes, and Edges

### What is a Graph in LangGraph?
A **graph** is the complete representation of an LLM workflow. It consists of **nodes** (tasks) and **edges** (control flow connections).

### Nodes
- Each node represents a **single task** (e.g., generate a topic, collect an essay, evaluate clarity, calculate score, give feedback).
- Behind the scenes, a node is just a **Python function** that takes the current **state** as input and returns an **updated state**.

### Edges
- Edges define **what runs next** after a node finishes.
- Types of edges:
  - **Sequential** – A → B → C (one after another).
  - **Parallel** – A splits into B, C, D running simultaneously.
  - **Conditional** – based on a condition, go to either X or Y.
  - **Looping** – go back to a previous node (creates a cycle).

**Example: UPSC Essay Evaluation Workflow**

```mermaid
graph TD
    A[Generate Topic] --> B[User Writes Essay]
    B --> C{Evaluate Essay}
    C --> D[Clarity Score]
    C --> E[Depth Score]
    C --> F[Language Score]
    D --> G[Aggregate Scores]
    E --> G
    F --> G
    G --> H{Total >= 10?}
    H -->|Yes| I[Congratulations]
    H -->|No| J[Provide Feedback]
    J --> K{Retry?}
    K -->|Yes| B
    K -->|No| L[End]
```

- **Nodes**: A, B, C, D, E, F, G, H, I, J, K, L.
- **Edges**: Sequential (A→B, B→C), Parallel (C→D, C→E, C→F), Conditional (H→I or H→J), Loop (K→B).

### Code Example (Conceptual)

```python
from langgraph.graph import StateGraph, END

# Define nodes as Python functions
def generate_topic(state):
    topic = llm.invoke("Generate a UPSC essay topic")
    return {**state, "topic": topic}

def collect_essay(state):
    essay = input("Write your essay:\n")
    return {**state, "essay": essay}

def evaluate_clarity(state):
    score = llm.invoke(f"Rate clarity of this essay (1-5):\n{state['essay']}")
    return {**state, "clarity_score": int(score)}

def evaluate_depth(state):
    score = llm.invoke(f"Rate depth of analysis (1-5):\n{state['essay']}")
    return {**state, "depth_score": int(score)}

def evaluate_language(state):
    score = llm.invoke(f"Rate language & grammar (1-5):\n{state['essay']}")
    return {**state, "language_score": int(score)}

def aggregate_scores(state):
    total = state["clarity_score"] + state["depth_score"] + state["language_score"]
    return {**state, "total_score": total}

def check_pass(state):
    return "pass" if state["total_score"] >= 10 else "fail"

def congratulate(state):
    print("Congratulations! You passed.")
    return state

def provide_feedback(state):
    feedback = llm.invoke(f"Provide feedback to improve this essay:\n{state['essay']}")
    print("Feedback:", feedback)
    return {**state, "feedback": feedback}

def ask_retry(state):
    retry = input("Try again? (y/n): ").lower() == 'y'
    return {**state, "retry": retry}

# Build graph
graph = StateGraph(State)
graph.add_node("generate_topic", generate_topic)
graph.add_node("collect_essay", collect_essay)
graph.add_node("evaluate_clarity", evaluate_clarity)
graph.add_node("evaluate_depth", evaluate_depth)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("aggregate_scores", aggregate_scores)
graph.add_node("congratulate", congratulate)
graph.add_node("provide_feedback", provide_feedback)
graph.add_node("ask_retry", ask_retry)

# Add edges
graph.set_entry_point("generate_topic")
graph.add_edge("generate_topic", "collect_essay")
graph.add_edge("collect_essay", "evaluate_clarity")
graph.add_edge("evaluate_clarity", "aggregate_scores")
graph.add_edge("evaluate_depth", "aggregate_scores")
graph.add_edge("evaluate_language", "aggregate_scores")
# Parallel edges: from aggregate_scores to three evaluators? Actually need to split.
# Better: add conditional routing after collecting essay.
```

**Key takeaway:** Nodes = tasks (Python functions). Edges = control flow (sequence, parallel, condition, loop).

---

## 2. State

### What is State?
**State** is the **shared memory** that flows through your workflow. It holds all the data that is needed during execution and that evolves over time.

In the UPSC example, state includes:
- `topic` (the essay topic)
- `essay` (the user’s written essay)
- `clarity_score`, `depth_score`, `language_score` (individual scores)
- `total_score` (aggregated)
- `feedback` (improvement suggestions)
- `retry` (boolean flag)

**Key properties of State:**
- **Shared** – every node can access the entire state.
- **Mutable** – any node can update any field in the state.
- **Evolves** – as execution progresses, state values change.

### How is State Implemented?
State is typically a **TypedDict** (a Python dictionary with type hints) or a Pydantic model.

```python
from typing import TypedDict

class EssayState(TypedDict):
    topic: str
    essay: str
    clarity_score: int
    depth_score: int
    language_score: int
    total_score: int
    feedback: str
    retry: bool
```

### State Flow in LangGraph

1. You provide an **initial state** (some fields may be empty).
2. The first node receives the state, does its work, and **returns an updated state**.
3. LangGraph automatically passes the updated state to the next node(s) via **edges**.
4. Each node can read and modify any field.

```python
# Example node that uses state
def evaluate_clarity(state: EssayState) -> EssayState:
    # Read current essay from state
    essay_text = state["essay"]
    # Compute score (e.g., call LLM)
    score = 4
    # Update state
    return {**state, "clarity_score": score}
```

✅ No manual passing of variables – LangGraph handles it.

---

## 3. Reducers

### Why Reducers?
By default, when a node updates a state key, it **replaces** the old value with the new one (`state["key"] = new_value`). This is fine for many cases (e.g., scores, flags). But sometimes you need different update logic:
- **Append** – for chat messages, you want to keep the whole conversation history, not just the last message.
- **Merge** – when multiple parallel nodes update different parts of a complex object.
- **Custom** – any other logic (e.g., taking the maximum, summing).

### Reducers in LangGraph
A **reducer** is a function that defines **how to apply an update** to a specific state key. Each key can have its own reducer.

Common reducers:
- `operator.add` – for appending to a list.
- `max` – keep the largest value.
- Custom function – e.g., merge two dictionaries.

### Example: Chatbot Conversation History

Without a reducer, the message history would be replaced each time, losing context.

```python
# Without reducer (bad for chat)
class ChatState(TypedDict):
    messages: str  # only the last message

def human_node(state):
    new_msg = input("You: ")
    return {**state, "messages": new_msg}   # previous message lost!

def llm_node(state):
    response = llm.invoke(state["messages"])
    return {**state, "messages": response}  # previous lost again
```

**With a reducer (append):**

```python
from typing import Annotated, List
from operator import add

class ChatState(TypedDict):
    messages: Annotated[List[str], add]   # reducer = add (appends)

def human_node(state: ChatState) -> ChatState:
    new_msg = input("You: ")
    return {"messages": [new_msg]}   # LangGraph appends this to existing list

def llm_node(state: ChatState) -> ChatState:
    response = llm.invoke(state["messages"][-1])  # last message
    return {"messages": [response]}  # appended

# After multiple turns, state["messages"] = [msg1, resp1, msg2, resp2, ...]
```

### Example: Parallel Updates Merging

When multiple nodes run in parallel and update the same state key, a reducer determines how to combine them.

```python
from typing import Annotated, List
from operator import add

class EvalState(TypedDict):
    all_scores: Annotated[List[int], add]   # reducer adds each new score

def eval_clarity(state): return {"all_scores": [4]}
def eval_depth(state):    return {"all_scores": [5]}
def eval_language(state): return {"all_scores": [3]}

# After parallel execution, state["all_scores"] = [4, 5, 3]
```

**Key takeaway:** Reducers give you fine‑grained control over state updates – essential for parallel workflows and history preservation.

---

## 4. LangGraph Execution Model (Inspired by Google Pregel)

LangGraph’s runtime is based on the **Pregel** model (used by Google for large‑scale graph processing). It follows these phases:

### Phase 1: Graph Definition
You define nodes, edges, and the state schema.

### Phase 2: Compilation
You call `.compile()` on the graph. LangGraph checks for structural errors (e.g., orphan nodes, unreachable nodes). If valid, it creates an executable graph object.

### Phase 3: Invocation
You call `.invoke(initial_state)` or `.stream()`. The graph starts execution from the **entry point** node.

### Phase 4: Message Passing
- A node receives the current state, executes its function, and returns **partial updates** to the state.
- These updates are **sent as messages** along the outgoing edges to the next node(s).
- If a node has multiple outgoing edges (parallel), the message is **copied** to all targets.

### Phase 5: Supersteps
A **superstep** is a round of execution that may contain **one or more parallel steps**.  
Why “superstep” and not “step”? Because a single round can involve multiple nodes executing simultaneously (e.g., three evaluators running in parallel). Each such round is a superstep.

- In a sequential graph, one superstep = one node.
- In a parallel graph, one superstep = multiple nodes running together.
- After all nodes in a superstep finish, their updates are **combined** using reducers, and the next superstep begins.

### Phase 6: Termination
Execution stops when:
- There are **no active nodes** (i.e., no node is currently executing), and
- There are **no messages in transit** (no pending updates to be sent).

When both conditions are true, the graph has reached a fixed point and terminates.

### Visualizing Supersteps (from the transcript)

```text
Superstep 1: generate_topic → collect_essay (sequential)
Superstep 2: evaluate_clarity, evaluate_depth, evaluate_language (parallel) → all run at once
Superstep 3: aggregate_scores
Superstep 4: conditional branch (congratulate or provide_feedback)
Superstep 5: ask_retry → if yes, jump back to collect_essay (another superstep)
```

### Code Example (Compilation & Invocation)

```python
# Define graph (as before)
graph = StateGraph(EssayState)
# ... add nodes and edges ...

# Compile
app = graph.compile()

# Initial state
initial = {
    "topic": "", "essay": "", "clarity_score": 0,
    "depth_score": 0, "language_score": 0,
    "total_score": 0, "feedback": "", "retry": False
}

# Invoke
final_state = app.invoke(initial)
print("Final state:", final_state)

# Or stream step by step (useful for debugging)
for step_output in app.stream(initial):
    print(step_output)
```

**Key takeaway:** You do not manually orchestrate nodes – LangGraph’s execution engine handles message passing and supersteps automatically.

---

## Summary Table

| Concept | What it is | Analogy |
|---------|------------|---------|
| **Graph** | The whole workflow | A blueprint of a factory |
| **Node** | A single task (Python function) | A machine in the factory |
| **Edge** | Control flow between tasks | Conveyor belt / routing |
| **State** | Shared data container | The product being worked on |
| **Reducer** | Update rule for state keys | “Append” vs “Replace” instructions |
| **Superstep** | One round of parallel/serial execution | A shift in the factory where multiple machines run |
| **Message Passing** | Sending state updates along edges | Passing the product to the next machine |

---

## Final Notes

- **Graphs, nodes, edges** are how you **design** workflows.
- **State** is how you **share data** across steps.
- **Reducers** give you control over **how state evolves** (replace, append, merge).
- **Execution model** (message passing, supersteps) is how LangGraph **runs** your workflow automatically.

---

## 06. Sequential Workflows in LangGraph (49:12)

## First Practical Workflow in LangGraph – BMI Calculator

This part of lecture marks the **start of the practical coding part**. After covering theory in the first four videos, the instructor walks through:

- **Installing LangGraph** and required libraries
- **Creating a virtual environment**
- **Building a simple sequential workflow** (BMI calculator) to understand:
  - Defining **State** with `TypedDict`
  - Creating **nodes** (as Python functions)
  - Adding **edges** (including `START` and `END`)
  - **Compiling** and **invoking** the graph
  - **Visualising** the graph in a Jupyter notebook
- **Extending** the workflow to a **two‑node sequential** workflow (calculate BMI → label category)

---

## 📌 Important Pointers

| # | Concept / Step | Explanation |
|---|----------------|--------------|
| 1 | **LangGraph is not a replacement for LangChain** | You use both together – LangChain provides components (LLMs, prompts, tools); LangGraph orchestrates them. |
| 2 | **Installation** | Create a virtual environment, then `pip install langgraph langchain langchain-openai python-dotenv`. |
| 3 | **Jupyter Notebook** | Recommended for development because you can **visualise graphs** easily (not possible in plain `.py` files). |
| 4 | **State** | A `TypedDict` that holds all data that flows through the workflow. Must be defined before building the graph. |
| 5 | **Nodes** | Each node is a **Python function** that receives the current state and returns an updated state. |
| 6 | **Edges** | Define the control flow. Special nodes: `START` (entry point) and `END` (termination). |
| 7 | **Compile** | `graph.compile()` checks the graph structure (no orphaned nodes, etc.) and returns a runnable object. |
| 8 | **Invoke** | `compiled_graph.invoke(initial_state)` executes the workflow and returns the final state. |
| 9 | **Visualisation** | Use `draw_mermaid` or helper code from LangGraph docs to see the graph as a diagram (Jupyter only). |
| 10 | **Sequential workflow** | Linear chain of nodes: A → B → C. No branching, no loops, no parallelism. |

---

## 1. Installation & Setup

### Create a virtual environment (Windows example)

```bash
python -m venv myenv
myenv\Scripts\activate
```

### Install required packages

```bash
pip install langgraph langchain langchain-openai python-dotenv
```

- **`langgraph`** – the orchestration framework
- **`langchain`** – provides components (models, prompts, retrievers, etc.)
- **`langchain-openai`** – OpenAI integration (used in later videos)
- **`python-dotenv`** – to read environment variables (e.g., API keys)

### Use Jupyter Notebook

```bash
jupyter notebook   # or use VS Code with Python Interactive
```

> **Why Jupyter?** LangGraph provides a way to **visualise the graph as a diagram** – this only works inside a Jupyter environment.

---

## 2. First Workflow: Single‑Node BMI Calculator

**Workflow description:**  
Input: weight (kg) and height (m) → calculate BMI → output BMI value.  
That’s one task → one node.

### Step 1: Imports

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
```

### Step 2: Define the State

The state must contain all data that will be passed between nodes.

```python
class BMIState(TypedDict):
    weight: float   # in kg
    height: float   # in meters
    bmi: float      # calculated value
```

**Explanation:**  
`TypedDict` is a special dictionary that lets you specify the **type** of each key. This helps with code completion and error checking.

### Step 3: Define the Node (as a Python function)

```python
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight"]
    height = state["height"]
    bmi = weight / (height ** 2)
    # Update the state with the new value
    state["bmi"] = round(bmi, 2)
    return state
```

**Key points:**
- The function receives the **current state**.
- It reads required fields (`weight`, `height`).
- It **modifies** the state (adds `bmi`).
- It returns the **updated state** (LangGraph will pass it to the next node).

### Step 4: Build the Graph

```python
# Create a graph that uses our state
graph = StateGraph(BMIState)

# Add the node – give it a name and the function
graph.add_node("calculate_bmi", calculate_bmi)

# Define edges: START → calculate_bmi → END
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", END)

# Set the entry point (alternative to START edge)
# graph.set_entry_point("calculate_bmi")  # works too
```

**Explanation:**
- `StateGraph(BMIState)` creates a new graph that understands the shape of the state.
- `add_node(name, function)` – every node must have a unique name.
- `START` and `END` are special built‑in nodes that mark the beginning and end of the graph.

### Step 5: Compile the Graph

```python
workflow = graph.compile()
```

**What happens during compilation?**  
LangGraph checks for structural errors (e.g., orphan nodes, cycles if not intended). If everything is fine, it returns a **runnable** object.

### Step 6: Invoke (Execute) the Workflow

```python
initial_state = {"weight": 80.0, "height": 1.73, "bmi": 0.0}
final_state = workflow.invoke(initial_state)
print(final_state)
```

**Output:**

```python
{'weight': 80.0, 'height': 1.73, 'bmi': 26.73}
```

> **Note:** The `bmi` field was `0.0` initially; after execution it contains the calculated value.

### Step 7: Visualise the Graph (Jupyter only)

```python
from IPython.display import Image, display
from langgraph.graph import draw_mermaid

display(Image(workflow.get_graph().draw_mermaid_png()))
```

**Result:** A picture showing `START → calculate_bmi → END`.

---

## 3. Extending to a Two‑Node Sequential Workflow

Add a second node that **classifies the BMI** into categories: underweight, normal, overweight, obese.

### Step 1: Update the State

Add a new field `category: str`.

```python
class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float
    category: str          # new field
```

### Step 2: Define the Second Node

```python
def label_bmi(state: BMIState) -> BMIState:
    bmi = state["bmi"]
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal"
    elif bmi < 30:
        category = "Overweight"
    else:
        category = "Obese"
    state["category"] = category
    return state
```

### Step 3: Add the Node and Edges to the Graph

```python
# Add the new node
graph.add_node("label_bmi", label_bmi)

# Sequential edges: START → calculate_bmi → label_bmi → END
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "label_bmi")
graph.add_edge("label_bmi", END)

# Compile again
workflow = graph.compile()
```

### Step 4: Run the Workflow

```python
initial_state = {"weight": 80.0, "height": 1.73, "bmi": 0.0, "category": ""}
final_state = workflow.invoke(initial_state)
print(final_state)
```

**Output:**

```python
{'weight': 80.0, 'height': 1.73, 'bmi': 26.73, 'category': 'Overweight'}
```

### Visualisation of the Two‑Node Graph

```
START → calculate_bmi → label_bmi → END
```

---

## 4. Complete Code Example (Copy‑Ready)

```python
# BMI Workflow with two nodes
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. Define state
class BMIState(TypedDict):
    weight: float
    height: float
    bmi: float
    category: str

# 2. Define nodes
def calculate_bmi(state: BMIState) -> BMIState:
    weight = state["weight"]
    height = state["height"]
    bmi = weight / (height ** 2)
    state["bmi"] = round(bmi, 2)
    return state

def label_bmi(state: BMIState) -> BMIState:
    bmi = state["bmi"]
    if bmi < 18.5:
        state["category"] = "Underweight"
    elif bmi < 25:
        state["category"] = "Normal"
    elif bmi < 30:
        state["category"] = "Overweight"
    else:
        state["category"] = "Obese"
    return state

# 3. Build graph
graph = StateGraph(BMIState)
graph.add_node("calculate_bmi", calculate_bmi)
graph.add_node("label_bmi", label_bmi)

# 4. Add edges (sequential)
graph.add_edge(START, "calculate_bmi")
graph.add_edge("calculate_bmi", "label_bmi")
graph.add_edge("label_bmi", END)

# 5. Compile
workflow = graph.compile()

# 6. Run
initial = {"weight": 80.0, "height": 1.73, "bmi": 0.0, "category": ""}
result = workflow.invoke(initial)
print(result)
```

---

## 5. Key Takeaways for Beginners

| Concept | What you did |
|---------|---------------|
| **State** | Defined a `TypedDict` with all fields that the workflow needs. |
| **Node** | Wrote a Python function that reads from state and writes updates. |
| **Graph** | Created a `StateGraph` passing the state class. |
| **Adding nodes** | Used `graph.add_node("name", function)`. |
| **Edges** | Connected nodes using `graph.add_edge(from, to)`. Used `START` and `END`. |
| **Compilation** | `graph.compile()` produces a runnable workflow. |
| **Invocation** | `workflow.invoke(initial_state)` runs the graph and returns the final state. |
| **Visualisation** | `workflow.get_graph().draw_mermaid_png()` shows the graph diagram (Jupyter only). |

---

## 6. Why This Matters

- This simple **sequential workflow** is the foundation of **all LangGraph applications**.
- Even complex agentic systems are built by adding more nodes, conditional edges, loops, and parallel branches – but **every graph starts with this basic pattern**.
- Once you master this, you can move to **prompt chaining**, **routing**, **parallelisation**, and finally **full agentic AI** where the LLM decides which node to run next.

---

## LLM Workflows in LangGraph – QA & Prompt Chaining 

This part of tutorial shows how to build **sequential LLM workflows** in LangGraph:

1. **Single‑node QA workflow** – ask an LLM a question, get an answer.
2. **Two‑node prompt chaining workflow** – generate an outline from a topic, then generate a full blog post from the outline.

The instructor also explains **why LangGraph is overkill for simple linear workflows** but essential when you need to **preserve intermediate state** (e.g., keeping the outline and the final blog together).

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **LLM workflow** | A workflow where at least one node calls an LLM. |
| 2 | **State for QA** | `question` (str) and `answer` (str). |
| 3 | **State for prompt chaining** | `topic`, `outline`, `content` (all strings). |
| 4 | **Node implementation** | Inside a node: extract data from state, build a prompt, call LLM, update state, return state. |
| 5 | **LangChain integration** | LangGraph nodes use LangChain components (`ChatOpenAI`, prompts, etc.). They work together seamlessly. |
| 6 | **Why preserve intermediate state?** | In LangChain chains, you lose intermediate outputs (only final result). In LangGraph, the state keeps everything – useful for debugging, auditing, or further processing. |
| 7 | **Homework** | Add a third node that **evaluates the blog** based on the outline and generates a score. |

---

## 1. Setup for LLM Workflows

Before building, you need:

- **OpenAI API key** stored in a `.env` file (or environment variable).
- Imports: `StateGraph`, `START`, `END`, `TypedDict`, `ChatOpenAI`, `load_dotenv`.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()                     # loads OPENAI_API_KEY from .env
model = ChatOpenAI(model="gpt-4")
```

---

## 2. Simple QA Workflow (One Node)

**Goal:** Ask a question → LLM answers → print answer.

### Step 1: Define State

```python
class LLMState(TypedDict):
    question: str
    answer: str
```

### Step 2: Define the Node (LLM call)

```python
def llm_qa(state: LLMState) -> LLMState:
    question = state["question"]
    prompt = f"Answer the following question:\n{question}"
    response = model.invoke(prompt)
    state["answer"] = response.content
    return state
```

### Step 3: Build Graph

```python
graph = StateGraph(LLMState)
graph.add_node("llm_qa", llm_qa)
graph.add_edge(START, "llm_qa")
graph.add_edge("llm_qa", END)
workflow = graph.compile()
```

### Step 4: Run

```python
initial_state = {"question": "How far is the moon from the earth?", "answer": ""}
final_state = workflow.invoke(initial_state)
print(final_state["answer"])
```

**Why this is still useful?** Because now you have both question and answer in the final state. In a simple direct call you would not keep the question.

---

## 3. Prompt Chaining Workflow (Two Nodes)

**Goal:**  
User gives a **topic** → Node 1 generates an **outline** → Node 2 generates a **full blog post** from the outline.

### Step 1: Define State

```python
class BlogState(TypedDict):
    topic: str
    outline: str
    content: str
```

### Step 2: Define Nodes

#### Node 1: Generate Outline

```python
def create_outline(state: BlogState) -> BlogState:
    topic = state["topic"]
    prompt = f"Generate a detailed outline for a blog on the topic: {topic}"
    response = model.invoke(prompt)
    state["outline"] = response.content
    return state
```

#### Node 2: Generate Blog from Outline

```python
def create_blog(state: BlogState) -> BlogState:
    topic = state["topic"]
    outline = state["outline"]
    prompt = f"Write a detailed blog on '{topic}' using this outline:\n{outline}"
    response = model.invoke(prompt)
    state["content"] = response.content
    return state
```

### Step 3: Build Graph

```python
graph = StateGraph(BlogState)
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)
graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)
workflow = graph.compile()
```

### Step 4: Run

```python
initial_state = {"topic": "Rise of AI in India", "outline": "", "content": ""}
final_state = workflow.invoke(initial_state)
print("Outline:\n", final_state["outline"])
print("\nBlog Content:\n", final_state["content"])
```

**Key advantage over LangChain chains:**  
- The final state contains **both the outline and the content** – not just the final output.  
- You can later add more nodes that use the outline or the content.

---

## 4. Why LangGraph Even for Simple Sequential Workflows?

For a single LLM call or a simple two‑step chain, **LangGraph is overkill** – you could just call the LLM directly.  
However, LangGraph forces you to use **state**, which means:

- All intermediate results are **preserved**.
- You can easily **insert new steps** (like evaluation) without breaking the flow.
- You can later add **conditional routing** (e.g., if the outline is too short, regenerate) without rewriting everything.

> *“It’s like going around your elbow to scratch your nose, but the real power appears when workflows become complex.”*

---

## 5. Homework Solution: Add an Evaluator Node

**Task:** Add a third node that evaluates the blog based on the outline and generates a score (integer 1‑10). Update the state to include the score.

### Step 1: Update State

Add a new field `score: int` (or `float`).

```python
class BlogState(TypedDict):
    topic: str
    outline: str
    content: str
    score: int               # new field
```

### Step 2: Define the Evaluator Node

```python
def evaluate_blog(state: BlogState) -> BlogState:
    outline = state["outline"]
    content = state["content"]
    prompt = f"""
    You are an evaluator. Given the outline and the final blog content, rate the blog on a scale of 1 to 10.
    Respond with only the integer score.

    Outline:
    {outline}

    Blog Content:
    {content}

    Score (1-10):
    """
    response = model.invoke(prompt)
    try:
        score = int(response.content.strip())
    except:
        score = 5   # default fallback
    state["score"] = score
    return state
```

### Step 3: Add Node and Edges

Insert the evaluator after `create_blog` and before `END`.

```python
graph.add_node("evaluate_blog", evaluate_blog)
# ... existing edges ...
graph.add_edge("create_blog", "evaluate_blog")   # chain: outline → blog → evaluate
graph.add_edge("evaluate_blog", END)
```

### Step 4: Run and See the Score

```python
initial_state = {"topic": "Rise of AI in India", "outline": "", "content": "", "score": 0}
final_state = workflow.invoke(initial_state)
print("Score:", final_state["score"])
```

**Full updated graph structure:**  
`START → create_outline → create_blog → evaluate_blog → END`

---

## 6. Complete Code for Prompt Chaining + Evaluation (Homework Solution)

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4")

class BlogState(TypedDict):
    topic: str
    outline: str
    content: str
    score: int

def create_outline(state: BlogState) -> BlogState:
    topic = state["topic"]
    prompt = f"Generate a detailed outline for a blog on: {topic}"
    response = model.invoke(prompt)
    state["outline"] = response.content
    return state

def create_blog(state: BlogState) -> BlogState:
    topic = state["topic"]
    outline = state["outline"]
    prompt = f"Write a detailed blog on '{topic}' using this outline:\n{outline}"
    response = model.invoke(prompt)
    state["content"] = response.content
    return state

def evaluate_blog(state: BlogState) -> BlogState:
    outline = state["outline"]
    content = state["content"]
    prompt = f"""
    Rate the blog (1-10) based on how well it follows the outline and quality.
    Outline: {outline}
    Blog: {content}
    Output only the integer score.
    """
    response = model.invoke(prompt)
    try:
        state["score"] = int(response.content.strip())
    except:
        state["score"] = 5
    return state

# Build graph
graph = StateGraph(BlogState)
graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)
graph.add_node("evaluate_blog", evaluate_blog)

graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", "evaluate_blog")
graph.add_edge("evaluate_blog", END)

workflow = graph.compile()

# Run
initial = {"topic": "Rise of AI in India", "outline": "", "content": "", "score": 0}
final = workflow.invoke(initial)
print(f"Score: {final['score']}/10")
# Print outline or content if needed
```

---

## 7. Key Takeaways

- **LangGraph nodes can contain any Python code** – including LLM calls, API calls, calculations, etc.
- **State is the central concept** – it passes through all nodes and accumulates data.
- **Sequential workflows** are built by adding simple edges (`START → A → B → END`).
- **LangChain components** (like `ChatOpenAI`) are used **inside** nodes.
- The real benefit of LangGraph appears when you add **branching, looping, human‑in‑the‑loop, or parallel execution** – which will be covered in upcoming videos.

---

## 07. Parallel Workflows in LangGraph (59:28)

This part of lecture shows how to build **parallel workflows** in LangGraph using two examples:

1. **Cricket statistics calculator** – a non‑LLM parallel workflow that calculates strike rate, boundary percentage, and balls per boundary from the same input data.
2. **Essay evaluation system** – an LLM‑based parallel workflow that evaluates an essay on three aspects (clarity, depth, language) simultaneously.

The key learning is **how to return partial state updates** instead of the full state when nodes run in parallel, to avoid update conflicts.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Parallel workflow** | Multiple nodes execute simultaneously from the same parent node. |
| 2 | **Use case** | When you have independent calculations on the same input data (e.g., different statistics, different evaluation criteria). |
| 3 | **The conflict problem** | If each parallel node returns the **entire updated state**, LangGraph sees conflicting updates to the same keys (e.g., `runs`, `balls`) and throws an `InvalidUpdateError`. |
| 4 | **Solution: Partial state updates** | Each node should return **only the keys it actually changed** (a dictionary with just those fields). LangGraph merges them automatically. |
| 5 | **Best practice** | Always return **partial state updates** (a dict of only the fields you modified) – works for both sequential and parallel workflows. |
| 6 | **Visualising parallel graphs** | Edges from `START` to multiple nodes create parallel execution. Use `add_edge(START, node1)`, `add_edge(START, node2)`, etc. |
| 7 | **Merging parallel results** | Add a final node that collects all parallel outputs (e.g., a `summary` node) and connects all parallel nodes to it. |

---

## 1. First Example: Cricket Stats Calculator (Non‑LLM Parallel Workflow)

**Goal:** Given runs, balls, fours, sixes, calculate:
- Strike rate = (runs / balls) × 100
- Boundary percentage = (4×fours + 6×sixes) / runs × 100
- Balls per boundary = balls / (fours + sixes)

These three calculations are **independent** → can run in parallel.

### Graph Structure

```
START ──┬──> calculate_strike_rate ──┐
        ├──> calculate_boundary_percent ──> summary_node ──> END
        └──> calculate_balls_per_boundary ──┘
```

### Step 1: Define State

```python
from typing import TypedDict

class BatsmanState(TypedDict):
    runs: int
    balls: int
    fours: int
    sixes: int
    strike_rate: float
    boundary_percent: float
    balls_per_boundary: float
    summary: str
```

### Step 2: Define Nodes (with **partial updates**)

```python
def calculate_strike_rate(state: BatsmanState):
    sr = (state["runs"] / state["balls"]) * 100
    # Return ONLY the field we changed
    return {"strike_rate": round(sr, 2)}

def calculate_boundary_percent(state: BatsmanState):
    boundary_runs = (state["fours"] * 4) + (state["sixes"] * 6)
    percent = (boundary_runs / state["runs"]) * 100
    return {"boundary_percent": round(percent, 2)}

def calculate_balls_per_boundary(state: BatsmanState):
    total_boundaries = state["fours"] + state["sixes"]
    bpb = state["balls"] / total_boundaries if total_boundaries > 0 else 0
    return {"balls_per_boundary": round(bpb, 2)}

def generate_summary(state: BatsmanState):
    summary = f"""
    Strike Rate: {state['strike_rate']}
    Boundary %: {state['boundary_percent']}
    Balls per Boundary: {state['balls_per_boundary']}
    """
    return {"summary": summary}
```

### Step 3: Build Graph with Parallel Edges

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(BatsmanState)
graph.add_node("calc_sr", calculate_strike_rate)
graph.add_node("calc_bp", calculate_boundary_percent)
graph.add_node("calc_bpb", calculate_balls_per_boundary)
graph.add_node("summary", generate_summary)

# Parallel edges from START to all three calculators
graph.add_edge(START, "calc_sr")
graph.add_edge(START, "calc_bp")
graph.add_edge(START, "calc_bpb")

# All three go to summary
graph.add_edge("calc_sr", "summary")
graph.add_edge("calc_bp", "summary")
graph.add_edge("calc_bpb", "summary")

graph.add_edge("summary", END)

workflow = graph.compile()
```

### Step 4: Run

```python
initial = {"runs": 100, "balls": 50, "fours": 6, "sixes": 4,
           "strike_rate": 0.0, "boundary_percent": 0.0,
           "balls_per_boundary": 0.0, "summary": ""}
final = workflow.invoke(initial)
print(final["summary"])
```

**Output:**
```
Strike Rate: 200.0
Boundary %: 48.0
Balls per Boundary: 5.0
```

---

## 2. The Parallel Update Error (and Why Partial Updates Fix It)

### What happens if you return the full state?

```python
# Wrong approach for parallel nodes
def calculate_strike_rate_wrong(state):
    sr = (state["runs"] / state["balls"]) * 100
    state["strike_rate"] = sr
    return state   # returns ALL fields
```

When three such nodes run in parallel, each returns the **entire state** (including `runs`, `balls`, etc.). LangGraph sees three conflicting updates to the same keys (e.g., `runs`) and throws:

```
InvalidUpdateError: At key 'runs' can receive only one value per step
```

### Why partial updates work

By returning **only the keys you changed**, you tell LangGraph: “I only modified these fields; leave the others as they are.” LangGraph merges the partial updates from parallel nodes without conflict.

```python
# Correct: return only the new field
return {"strike_rate": sr}
```

**Best practice:** Always use partial updates – they work in sequential workflows too.

---

## 3. Second Example: LLM‑Based Parallel Essay Evaluation

**Goal:** Evaluate a UPSC essay on three aspects in parallel:
- Clarity of thought
- Depth of analysis
- Language quality

Each evaluation produces:
- A textual **feedback**
- A **score** (0‑10)

All three run simultaneously, then a final node aggregates the results.

### Graph Structure (same parallel pattern)

```
START ──┬──> evaluate_clarity ──┐
        ├──> evaluate_depth ────┼──> aggregate_results ──> END
        └──> evaluate_language ─┘
```

### Step 1: Define State

```python
class EssayState(TypedDict):
    essay: str
    clarity_feedback: str
    clarity_score: float
    depth_feedback: str
    depth_score: float
    language_feedback: str
    language_score: float
    final_report: str
```

### Step 2: Define Nodes (each calls LLM)

```python
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4")

def evaluate_clarity(state: EssayState):
    prompt = f"Evaluate clarity of thought in this essay (0-10). Return JSON: {{'score': int, 'feedback': 'text'}}\nEssay: {state['essay']}"
    response = model.invoke(prompt)
    # Parse response (simplified)
    import json
    data = json.loads(response.content)
    return {"clarity_score": data["score"], "clarity_feedback": data["feedback"]}

def evaluate_depth(state: EssayState):
    prompt = f"Evaluate depth of analysis in this essay (0-10). Return JSON...\nEssay: {state['essay']}"
    response = model.invoke(prompt)
    data = json.loads(response.content)
    return {"depth_score": data["score"], "depth_feedback": data["feedback"]}

def evaluate_language(state: EssayState):
    prompt = f"Evaluate language quality in this essay (0-10). Return JSON...\nEssay: {state['essay']}"
    response = model.invoke(prompt)
    data = json.loads(response.content)
    return {"language_score": data["score"], "language_feedback": data["feedback"]}

def aggregate_results(state: EssayState):
    report = f"""
    CLARITY: {state['clarity_score']}/10 - {state['clarity_feedback']}
    DEPTH: {state['depth_score']}/10 - {state['depth_feedback']}
    LANGUAGE: {state['language_score']}/10 - {state['language_feedback']}
    """
    return {"final_report": report}
```

### Step 3: Build Graph

```python
graph = StateGraph(EssayState)
graph.add_node("clarity", evaluate_clarity)
graph.add_node("depth", evaluate_depth)
graph.add_node("language", evaluate_language)
graph.add_node("aggregate", aggregate_results)

# Parallel edges
graph.add_edge(START, "clarity")
graph.add_edge(START, "depth")
graph.add_edge(START, "language")

# All go to aggregate
graph.add_edge("clarity", "aggregate")
graph.add_edge("depth", "aggregate")
graph.add_edge("language", "aggregate")
graph.add_edge("aggregate", END)

workflow = graph.compile()
```

### Step 4: Run

```python
initial = {"essay": "Democracy in India has evolved...", 
           "clarity_feedback": "", "clarity_score": 0.0,
           "depth_feedback": "", "depth_score": 0.0,
           "language_feedback": "", "language_score": 0.0,
           "final_report": ""}
final = workflow.invoke(initial)
print(final["final_report"])
```

---

## 4. Summary: Parallel Workflow Patterns

| Aspect | Sequential (previous video) | Parallel (this video) |
|--------|----------------------------|----------------------|
| **Edges** | Single chain START→A→B→END | Multiple edges from START to different nodes |
| **Node return** | Full state or partial (both work) | **Must use partial updates** (only changed keys) |
| **Use case** | Step‑by‑step pipeline | Independent calculations on same input |
| **Example** | Topic → outline → blog | Strike rate + boundary% + balls/boundary |

---

## 5. Key Takeaways

- **Parallel execution** is achieved by adding multiple edges from the same source node (usually `START`).
- **Always return partial state updates** in parallel nodes – a dictionary containing only the fields you modified.
- LangGraph automatically merges partial updates from parallel nodes.
- Parallel workflows are ideal for **independent evaluations** (like essay scoring) or **multiple metrics** from the same data.
- The same parallel pattern can be used with or without LLMs.

---

## LLM‑Based Parallel Workflow with Structured Output & Reducers

This part of tutorial builds a **real‑world parallel LLM workflow** – an **UPSC essay evaluation system**. The workflow:

1. Takes an essay as input.
2. Spawns **three parallel nodes**, each using an LLM to evaluate a different aspect:
   - **Clarity of thought**
   - **Depth of analysis**
   - **Language quality**
3. Each node returns a **feedback text** and a **score (0‑10)**.
4. A final node **aggregates** the results:
   - Merges the three feedback texts into a **summarised feedback** (using an LLM).
   - Calculates the **average score**.
5. Outputs the summary and average score.

The tutorial emphasises three advanced concepts:
- **Structured output** (Pydantic schema) to force the LLM to return reliable JSON.
- **Reducers** (specifically `operator.add`) to merge scores from parallel nodes into a list without overwriting.
- **Parallel node design** with partial state updates.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Parallel LLM evaluation** | The same essay is sent to three independent LLM nodes – they run simultaneously. |
| 2 | **Structured output** | Use `ChatOpenAI` with `.with_structured_output(schema)` to force the LLM to return a predictable JSON format (Pydantic model). |
| 3 | **Pydantic schema** | Define a class with fields and descriptions to guide the LLM. |
| 4 | **Reducer** | A function that tells LangGraph how to combine multiple updates to the **same state key** from parallel nodes. |
| 5 | **`operator.add`** | A built‑in reducer that **concatenates lists** – used to merge individual scores into a list. |
| 6 | **`Annotated` type** | Wrapping a state field with `Annotated[list, operator.add]` tells LangGraph to use that reducer. |
| 7 | **Partial updates** | Each node returns a dictionary containing **only the fields it changed** – essential for parallel execution to avoid conflicts. |
| 8 | **Final aggregation node** | Collects the parallel outputs, computes average, and generates a combined feedback using a normal (non‑structured) LLM. |

---

## 1. Structured Output – Forcing LLM to Return Reliable Data

### Problem
When you ask an LLM for a score, it might return `"7"`, `"seven"`, or `"7.0"`. You need a consistent, parseable format.

### Solution: Pydantic Schema + `with_structured_output`

```python
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Define the expected output shape
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", ge=0, le=10)

# Create a model that always returns this structure
model = ChatOpenAI(model="gpt-4o-mini")
structured_model = model.with_structured_output(EvaluationSchema)

# Use it
essay = "India has many smart students..."
prompt = f"Evaluate the language quality of this essay:\n{essay}"
result = structured_model.invoke(prompt)
print(result.feedback)   # string
print(result.score)      # int, always between 0 and 10
```

**Why this works:**  
The LLM is forced to output JSON matching the schema. LangChain handles parsing automatically. The `Field` descriptions and constraints (`ge=0, le=10`) help the LLM produce correct values.

---

## 2. State Definition with a Reducer

We need a state key `individual_scores` that holds **all three scores** (from the three parallel nodes). Because the nodes run at the same time, each tries to update the same key. Without a reducer, the last update would overwrite the previous ones.

**Solution:** Use `Annotated` with `operator.add` as the reducer.

```python
from typing import TypedDict, Annotated, List
import operator

class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add]   # reducer = add
    average_score: float
```

**How `operator.add` works:**
- Each node returns a **list with one score**, e.g., `{"individual_scores": [7]}`.
- The reducer **concatenates** all such lists into one: `[7] + [8] + [6] = [7, 8, 6]`.
- No overwriting – perfect for parallel nodes.

---

## 3. Parallel Node Implementation

Each node follows the same pattern:
- Read the essay from state.
- Build a prompt specific to the aspect (language, depth, clarity).
- Call the **structured model**.
- Return a **partial update** (only the fields it changed).

```python
def evaluate_language(state: UPSCState) -> dict:
    prompt = f"Evaluate the language quality of this essay. Give feedback and a score out of 10.\nEssay: {state['essay']}"
    result = structured_model.invoke(prompt)
    return {
        "language_feedback": result.feedback,
        "individual_scores": [result.score]   # note: list with one element
    }

def evaluate_analysis(state: UPSCState) -> dict:
    prompt = f"Evaluate the depth of analysis in this essay...\nEssay: {state['essay']}"
    result = structured_model.invoke(prompt)
    return {
        "analysis_feedback": result.feedback,
        "individual_scores": [result.score]
    }

def evaluate_clarity(state: UPSCState) -> dict:
    prompt = f"Evaluate the clarity of thought in this essay...\nEssay: {state['essay']}"
    result = structured_model.invoke(prompt)
    return {
        "clarity_feedback": result.feedback,
        "individual_scores": [result.score]
    }
```

**Important:** Each returns `"individual_scores": [score]` (a list with one int). The reducer will merge them.

---

## 4. Final Aggregation Node

This node runs **after** all three parallel nodes finish. It:
- Generates a **summarised feedback** from the three individual feedbacks (using a normal LLM).
- Calculates the average score from the list.

```python
def final_evaluation(state: UPSCState) -> dict:
    # 1. Summarise feedback using a normal (non-structured) LLM
    normal_model = ChatOpenAI(model="gpt-4o-mini")
    prompt = f"""
    Based on the following feedbacks, create a single summarised feedback for the essay.
    Language feedback: {state['language_feedback']}
    Depth feedback: {state['analysis_feedback']}
    Clarity feedback: {state['clarity_feedback']}
    """
    summary_response = normal_model.invoke(prompt)
    overall_feedback = summary_response.content

    # 2. Calculate average score
    scores = state["individual_scores"]   # e.g., [7, 8, 6]
    average = sum(scores) / len(scores)

    return {
        "overall_feedback": overall_feedback,
        "average_score": round(average, 2)
    }
```

---

## 5. Building the Graph with Parallel Edges

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(UPSCState)

# Add nodes
graph.add_node("eval_language", evaluate_language)
graph.add_node("eval_analysis", evaluate_analysis)
graph.add_node("eval_clarity", evaluate_clarity)
graph.add_node("final_eval", final_evaluation)

# Parallel edges from START to all three evaluators
graph.add_edge(START, "eval_language")
graph.add_edge(START, "eval_analysis")
graph.add_edge(START, "eval_clarity")

# All three go to final_eval
graph.add_edge("eval_language", "final_eval")
graph.add_edge("eval_analysis", "final_eval")
graph.add_edge("eval_clarity", "final_eval")

graph.add_edge("final_eval", END)

workflow = graph.compile()
```

---

## 6. Running the Workflow

```python
initial_state = {
    "essay": "India has many smart students and engineers...",
    "language_feedback": "",
    "analysis_feedback": "",
    "clarity_feedback": "",
    "overall_feedback": "",
    "individual_scores": [],
    "average_score": 0.0
}

final_state = workflow.invoke(initial_state)
print(f"Individual scores: {final_state['individual_scores']}")
print(f"Average score: {final_state['average_score']}")
print(f"Overall feedback: {final_state['overall_feedback']}")
```

**Sample output:**
```
Individual scores: [7, 8, 8]
Average score: 7.67
Overall feedback: The essay shows good language skills and depth, but clarity could be improved...
```

---

## 7. Why Reducers Are Essential in Parallel Workflows

| Without reducer | With reducer (`operator.add`) |
|----------------|-------------------------------|
| Last node wins → only one score stored | All scores merged into a list |
| Data loss | Complete data preserved |
| Cannot compute average | Can compute average easily |

**Under the hood:**  
LangGraph detects that the `individual_scores` field has a reducer. When multiple nodes return partial updates to the same key, it calls the reducer function (here `operator.add`) with the existing value and the new value, and stores the result.

---

## 8. Summary: Parallel LLM Workflow Patterns

| Concept | Implementation |
|---------|----------------|
| **Parallel execution** | Multiple `add_edge(START, node)` |
| **Structured output** | `model.with_structured_output(PydanticModel)` |
| **Merging parallel results** | `Annotated[List, operator.add]` in state |
| **Partial updates** | Each node returns dict with only changed keys |
| **Final aggregation** | A node that reads all parallel outputs and computes final result |

---

## 9. Key Takeaways

- **Always use partial updates** in parallel nodes (return a dict with only the fields you modified).
- **Structured output** (Pydantic) is essential when you need reliable, parseable data from LLMs – especially numbers and specific formats.
- **Reducers** (`operator.add`, `operator.mul`, `max`, or custom functions) give you fine‑grained control over how parallel updates are merged.
- The **final aggregation node** is where you combine all parallel results – it runs only after all parallel nodes finish.
- LangChain and LangGraph work **hand‑in‑hand** – LangChain for LLM interaction and structured output, LangGraph for orchestration and state management.

---

## 08. Conditional Workflows in LangGraph (47:37)

This part of tutorial introduces **conditional workflows** – the third type of workflow after sequential and parallel. Conditional workflows are like **if‑else statements** for graphs: based on a condition, you choose **only one** of several possible paths to execute, not all of them simultaneously.  

The instructor builds a **quadratic equation solver** as a non‑LLM example to demonstrate conditional branching, then mentions a follow‑up LLM‑based customer support example.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Conditional workflow** | A graph where, after a node, the next node is chosen dynamically based on a condition (like `if‑elif‑else`). Only one branch executes. |
| 2 | **Router function** | A Python function that takes the current `state` and **returns the name of the next node** as a string. |
| 3 | **`add_conditional_edges`** | LangGraph method that adds conditional edges from a source node. It takes: source node, router function, and (optionally) a mapping of return values to node names. |
| 4 | **No parallel execution** | Unlike parallel workflows, conditional workflows execute **only one branch** per run. |
| 5 | **Quadratic equation solver** | A perfect real‑world example because the discriminant determines three distinct cases (two real roots, one repeated root, no real roots). |

---

## 1. Recap of Previous Workflow Types

- **Sequential** – linear chain (A → B → C).  
- **Parallel** – multiple branches execute simultaneously (A → B and A → C at the same time).  
- **Conditional** – multiple branches are available, but only **one** is chosen based on a condition (A → (if condition) B else C).

---

## 2. Quadratic Equation Solver – A Conditional Workflow

### Mathematical Background

For a quadratic equation: `ax² + bx + c = 0`

- **Discriminant** `D = b² - 4ac`
- If `D > 0` → two distinct real roots  
  `root1 = (-b + sqrt(D)) / (2a)`  
  `root2 = (-b - sqrt(D)) / (2a)`
- If `D == 0` → one repeated real root  
  `root = -b / (2a)`
- If `D < 0` → no real roots

We will build a graph that:
1. **Displays the equation** (show a, b, c).
2. **Calculates discriminant**.
3. **Branches conditionally** to the appropriate root‑calculation node.
4. **Returns the result**.

### Graph Structure

```
START → show_equation → calculate_discriminant
                              |
                              ↓ (router based on D)
                ┌─────────────┼─────────────┐
                ↓             ↓             ↓
          real_roots   repeated_root   no_real_roots
                ↓             ↓             ↓
                └─────────────┴─────────────┘
                              ↓
                             END
```

---

## 3. Step‑by‑Step Code Example

### 3.1 Imports and State Definition

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class QuadState(TypedDict):
    a: float
    b: float
    c: float
    equation: str
    discriminant: float
    result: str
```

### 3.2 Node Functions

#### Node 1: Show the equation

```python
def show_equation(state: QuadState) -> dict:
    eq = f"{state['a']}x² + {state['b']}x + {state['c']}"
    return {"equation": eq}
```

#### Node 2: Calculate discriminant

```python
def calculate_discriminant(state: QuadState) -> dict:
    d = (state['b'] ** 2) - (4 * state['a'] * state['c'])
    return {"discriminant": d}
```

#### Node 3: Two real roots

```python
import math

def real_roots(state: QuadState) -> dict:
    a, b, d = state['a'], state['b'], state['discriminant']
    root1 = (-b + math.sqrt(d)) / (2 * a)
    root2 = (-b - math.sqrt(d)) / (2 * a)
    result = f"Two real roots: {root1:.2f} and {root2:.2f}"
    return {"result": result}
```

#### Node 4: One repeated root

```python
def repeated_root(state: QuadState) -> dict:
    a, b = state['a'], state['b']
    root = -b / (2 * a)
    result = f"One repeated root: {root:.2f}"
    return {"result": result}
```

#### Node 5: No real roots

```python
def no_real_roots(state: QuadState) -> dict:
    result = "No real roots (discriminant < 0)"
    return {"result": result}
```

### 3.3 Router Function (Condition)

```python
def route_condition(state: QuadState) -> str:
    d = state["discriminant"]
    if d > 0:
        return "real_roots"
    elif d == 0:
        return "repeated_root"
    else:
        return "no_real_roots"
```

**Important:** This function returns the **name of the next node** as a string, not the node object.

### 3.4 Building the Graph with Conditional Edges

```python
# Create graph
graph = StateGraph(QuadState)

# Add all nodes
graph.add_node("show_equation", show_equation)
graph.add_node("calculate_discriminant", calculate_discriminant)
graph.add_node("real_roots", real_roots)
graph.add_node("repeated_root", repeated_root)
graph.add_node("no_real_roots", no_real_roots)

# Sequential edges (unconditional)
graph.add_edge(START, "show_equation")
graph.add_edge("show_equation", "calculate_discriminant")

# Conditional edges from calculate_discriminant
graph.add_conditional_edges(
    "calculate_discriminant",          # source node
    route_condition,                   # router function
    {
        "real_roots": "real_roots",
        "repeated_root": "repeated_root",
        "no_real_roots": "no_real_roots"
    }
)

# All three result nodes go to END
graph.add_edge("real_roots", END)
graph.add_edge("repeated_root", END)
graph.add_edge("no_real_roots", END)

# Compile
workflow = graph.compile()
```

### 3.5 Running the Workflow

```python
# Case 1: D > 0 (a=4, b=-5, c=-4)
initial_state = {"a": 4, "b": -5, "c": -4, "equation": "", "discriminant": 0.0, "result": ""}
final = workflow.invoke(initial_state)
print(final["equation"])      # 4x² + -5x + -4
print(final["discriminant"])  # 89.0
print(final["result"])        # Two real roots: 1.56 and -0.56

# Case 2: D == 0 (a=1, b=2, c=1)
initial_state2 = {"a": 1, "b": 2, "c": 1, ...}
final2 = workflow.invoke(initial_state2)
print(final2["result"])       # One repeated root: -1.00

# Case 3: D < 0 (a=1, b=2, c=5)
initial_state3 = {"a": 1, "b": 2, "c": 5, ...}
final3 = workflow.invoke(initial_state3)
print(final3["result"])       # No real roots (discriminant < 0)
```

---

## 4. Visualising the Conditional Graph

When you visualise the compiled graph, conditional edges are shown as **dotted lines** – meaning only one of them will be taken per run.

```python
from IPython.display import Image, display
display(Image(workflow.get_graph().draw_mermaid_png()))
```

You will see:  
`START → show_equation → calculate_discriminant`  
Then three dotted arrows to `real_roots`, `repeated_root`, `no_real_roots`.  
Finally, solid arrows from each to `END`.

---

## 5. Key Takeaways

- **`add_conditional_edges`** replaces `add_edge` when the next node depends on a runtime condition.
- The **router function** must return a string matching one of the node names.
- The optional third argument (a dictionary) maps return values to node names – useful when the router returns codes like `"case1"` instead of the actual node name.
- Conditional workflows are **deterministic** for a given input – only one path is taken.
- This pattern is essential for building **agentic behaviour**, where the LLM decides which tool or sub‑graph to call next (will be covered in later videos).

---

## Customer Review Handler

This part tutorial builds a **real‑world conditional workflow** using LangGraph: a customer review response system.  
The workflow:

1. Takes a customer **review** (text).
2. Uses an LLM to classify **sentiment** (positive / negative) – structured output.
3. **Conditionally branches**:
   - If **positive** → generates a warm thank‑you message.
   - If **negative** → runs a **diagnosis** (extracts issue type, tone, urgency) → then generates an empathetic resolution message.
4. Outputs the final response.

This demonstrates **conditional edges** – the graph chooses exactly one path based on the sentiment.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Sentiment classification** | First LLM call decides if review is positive or negative (structured output: `{"sentiment": "positive" or "negative"}`). |
| 2 | **Conditional branch** | Based on the sentiment, the graph routes to either a positive response node or a diagnosis node. |
| 3 | **Diagnosis** | For negative reviews, a second structured LLM call extracts `issue_type`, `tone`, `urgency`. |
| 4 | **Response generation** | Two different LLM calls generate the final reply: one for positive, one for negative (using the diagnosis data). |
| 5 | **State definition** | Contains `review`, `sentiment`, `diagnosis` (dict), and `response`. |
| 6 | **Two structured schemas** | `SentimentSchema` (one field) and `DiagnosisSchema` (three fields). |
| 7 | **Router function** | `check_sentiment(state)` returns `"positive_response"` or `"run_diagnosis"` – the name of the next node. |
| 8 | **`add_conditional_edges`** | Links the sentiment node to the router function; only one outgoing edge is taken per run. |

---

## 1. Workflow Diagram

```
START → find_sentiment
              ↓
        (router based on sentiment)
         ↙         ↘
 positive_response   run_diagnosis
        ↓                  ↓
        END          negative_response
                           ↓
                          END
```

---

## 2. State Definition

The state holds all data that flows through the graph.

```python
from typing import TypedDict, Dict

class ReviewState(TypedDict):
    review: str
    sentiment: str          # "positive" or "negative"
    diagnosis: Dict[str, str]  # {issue_type, tone, urgency}
    response: str
```

---

## 3. Structured Output Schemas (Pydantic)

We need **two schemas** – one for sentiment, one for diagnosis.

### Sentiment Schema

```python
from pydantic import BaseModel, Field
from typing import Literal

class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Sentiment of the review")
```

### Diagnosis Schema

```python
class DiagnosisSchema(BaseModel):
    issue_type: Literal["ui", "performance", "bug", "support", "other"] = Field(description="Category of the issue")
    tone: Literal["frustrated", "angry", "disappointed", "neutral"] = Field(description="Emotional tone of the user")
    urgency: Literal["low", "medium", "high"] = Field(description="How urgent the issue appears")
```

---

## 4. Creating Structured LLM Models

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini")

# Model for sentiment
sentiment_model = model.with_structured_output(SentimentSchema)

# Model for diagnosis
diagnosis_model = model.with_structured_output(DiagnosisSchema)
```

We also keep a **normal (unstructured) model** for generating free‑text responses.

---

## 5. Node Functions

### Node 1: Find Sentiment

```python
def find_sentiment(state: ReviewState) -> dict:
    prompt = f"What is the sentiment of this review? {state['review']}"
    result = sentiment_model.invoke(prompt)
    return {"sentiment": result.sentiment}
```

### Node 2: Positive Response (generates thank‑you message)

```python
def positive_response(state: ReviewState) -> dict:
    prompt = f"Write a warm thank-you message in response to this review: {state['review']}. Also kindly ask the user to leave feedback on our website."
    response = model.invoke(prompt)
    return {"response": response.content}
```

### Node 3: Run Diagnosis (structured extraction)

```python
def run_diagnosis(state: ReviewState) -> dict:
    prompt = f"Diagnose this negative review. Return issue type, tone, and urgency. Review: {state['review']}"
    result = diagnosis_model.invoke(prompt)
    # Convert Pydantic model to dict
    return {"diagnosis": result.model_dump()}
```

### Node 4: Negative Response (empathetic resolution)

```python
def negative_response(state: ReviewState) -> dict:
    diagnosis = state["diagnosis"]
    prompt = f"""
    You are a support assistant. The user had an issue: {diagnosis['issue_type']}.
    Their tone was: {diagnosis['tone']}. Urgency: {diagnosis['urgency']}.
    Write an empathetic, helpful resolution message.
    """
    response = model.invoke(prompt)
    return {"response": response.content}
```

---

## 6. Router Function (Condition)

```python
def check_sentiment(state: ReviewState) -> str:
    if state["sentiment"] == "positive":
        return "positive_response"
    else:
        return "run_diagnosis"
```

**Returns the name of the next node** as a string.

---

## 7. Building the Graph with Conditional Edges

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(ReviewState)

# Add nodes
graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)

# Sequential edges
graph.add_edge(START, "find_sentiment")

# Conditional edge from find_sentiment
graph.add_conditional_edges(
    "find_sentiment",           # source node
    check_sentiment,            # router function
    {
        "positive_response": "positive_response",
        "run_diagnosis": "run_diagnosis"
    }
)

# Edges from the two branches
graph.add_edge("positive_response", END)
graph.add_edge("run_diagnosis", "negative_response")
graph.add_edge("negative_response", END)

workflow = graph.compile()
```

---

## 8. Running the Workflow

### Example 1: Positive Review

```python
initial = {
    "review": "I've been using this app for a month. The UI is incredibly clean and intuitive. Great job!",
    "sentiment": "",
    "diagnosis": {},
    "response": ""
}
final = workflow.invoke(initial)
print(final["response"])
```

**Output (sample):**  
*"Thank you for your kind words! We're thrilled you enjoy the app. Please consider leaving a review on our website!"*

### Example 2: Negative Review

```python
initial = {
    "review": "I've been trying to log in for over an hour. The app keeps freezing on the authentication screen. This bug is unacceptable.",
    "sentiment": "",
    "diagnosis": {},
    "response": ""
}
final = workflow.invoke(initial)
print(final["diagnosis"])   # {'issue_type': 'bug', 'tone': 'frustrated', 'urgency': 'high'}
print(final["response"])
```

**Output (sample):**  
*"We're really sorry you're experiencing login issues. Our team is investigating the bug you reported. Please contact support at help@example.com so we can resolve this urgently."*

---

## 9. Key Takeaways

- **Conditional workflows** are implemented using `add_conditional_edges` + a router function that returns the **next node name**.
- The router function can use any logic – here it simply checks the `sentiment` field from the state.
- Structured output (Pydantic) ensures reliable parsing of sentiment and diagnosis data.
- Multiple structured models can be used for different tasks (sentiment, diagnosis).
- The final response uses **unstructured** LLM calls to generate natural, empathetic language.

---

## 10. Comparison: Conditional vs Parallel Workflows

| Feature | Conditional Workflow | Parallel Workflow |
|---------|----------------------|--------------------|
| **Edges** | `add_conditional_edges` | multiple `add_edge` from same source |
| **Execution** | Only one branch runs | All branches run simultaneously |
| **Router needed?** | Yes – returns node name | No – all edges are taken |
| **Use case** | If‑else decision | Independent calculations |

---

## 09. Iterative Workflows in LangGraph (37:13)

## Iterative / Looping Workflow – Tweet Generator with Evaluation & Optimization

This part of tutorial builds the **fourth type** of workflow after sequential, parallel, and conditional: an **iterative (looping) workflow**.  
The goal is to automatically generate a **funny, original tweet** on a given topic, evaluate it against strict criteria, and if it fails, **optimize** it and loop back until it passes or a maximum iteration limit is reached.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Iterative workflow** | A loop where a task is repeatedly improved based on feedback until a condition is met (like “approved”). |
| 2 | **Three LLM roles** | **Generator** – creates initial/improved tweet. **Evaluator** – judges tweet (approved / needs improvement) + gives feedback. **Optimizer** – rewrites tweet using feedback. |
| 3 | **Structured output** | Evaluator returns a fixed schema (`evaluation` and `feedback`) using Pydantic and `.with_structured_output()`. |
| 4 | **Reducers** | Used to accumulate `tweet_history` and `feedback_history` across iterations (with `operator.add`). |
| 5 | **Conditional edge + loop edge** | From `evaluate` node: conditional edge to `END` if approved, else to `optimize`. Then a normal edge from `optimize` back to `evaluate` – forms the loop. |
| 6 | **Max iterations** | Prevents infinite loops. If `iteration >= max_iterations` and still not approved, force approval and exit. |
| 7 | **State design** | Holds topic, current tweet, evaluation status, feedback, iteration counter, max iterations, and history lists. |

---

## 1. Workflow Diagram

```
START → generate → evaluate
                      ↓ (conditional)
        ┌─────────────┴─────────────┐
        ↓ (approved)                ↓ (needs improvement)
        END                       optimize
                                     ↓ (normal edge)
                                  evaluate (loop back)
```

**Key point:** The edge from `optimize` back to `evaluate` creates a **loop** – the graph will keep cycling until the tweet is approved or max iterations reached.

---

## 2. State Definition with Reducers for History

We want to keep track of **all generated tweets** and **all feedbacks** across iterations.  
Use `Annotated` with `operator.add` as a reducer to **append** new items to a list.

```python
from typing import TypedDict, Annotated, List
import operator

class TweetState(TypedDict):
    topic: str
    tweet: str
    evaluation: str   # "approved" or "needs_improvement"
    feedback: str
    iteration: int
    max_iterations: int
    tweet_history: Annotated[List[str], operator.add]
    feedback_history: Annotated[List[str], operator.add]
```

- `tweet_history` and `feedback_history` use `operator.add` – each new value is **appended** (merged) into the list, not overwritten.

---

## 3. Creating the Three LLMs (Generator, Evaluator, Optimizer)

```python
from langchain_openai import ChatOpenAI

# Generator: strong writing model
generator_llm = ChatOpenAI(model="gpt-4o")

# Evaluator: structured output model (with Pydantic schema)
evaluator_llm = ChatOpenAI(model="gpt-4o-mini")

# Optimizer: model to rewrite tweets
optimizer_llm = ChatOpenAI(model="gpt-4o")
```

### 3.1 Evaluator Structured Output Schema

```python
from pydantic import BaseModel, Field
from typing import Literal

class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(description="Whether the tweet passes the criteria")
    feedback: str = Field(description="Constructive feedback to improve the tweet")

structured_evaluator = evaluator_llm.with_structured_output(TweetEvaluation)
```

---

## 4. Node Functions

### 4.1 Generate Tweet (Generator Node)

```python
def generate_tweet(state: TweetState) -> dict:
    prompt = f"You are a funny Twitter influencer. Write a short, original, hilarious tweet about: {state['topic']}."
    response = generator_llm.invoke(prompt)
    new_tweet = response.content
    return {"tweet": new_tweet, "tweet_history": [new_tweet]}
```

### 4.2 Evaluate Tweet (Evaluator Node)

```python
def evaluate_tweet(state: TweetState) -> dict:
    messages = [
        ("system", "You are a ruthless tweet critic. Evaluate based on originality, humor, virality, format."),
        ("human", f"Tweet: {state['tweet']}\n\nEvaluate (approved/needs_improvement) and provide feedback.")
    ]
    result = structured_evaluator.invoke(messages)
    return {
        "evaluation": result.evaluation,
        "feedback": result.feedback,
        "feedback_history": [result.feedback]
    }
```

### 4.3 Optimize Tweet (Optimizer Node)

```python
def optimize_tweet(state: TweetState) -> dict:
    messages = [
        ("system", "You punch up tweets for virality and humor based on feedback."),
        ("human", f"""
        Improve the tweet based on this feedback.
        Topic: {state['topic']}
        Original tweet: {state['tweet']}
        Feedback: {state['feedback']}
        Write a short, viral, funny tweet (avoid Q&A style, under 280 chars).
        """)
    ]
    response = optimizer_llm.invoke(messages)
    improved_tweet = response.content
    return {
        "tweet": improved_tweet,
        "iteration": state["iteration"] + 1,
        "tweet_history": [improved_tweet]
    }
```

---

## 5. Router Function for Conditional Edge

```python
def route_evaluation(state: TweetState) -> str:
    if state["evaluation"] == "approved" or state["iteration"] >= state["max_iterations"]:
        return "approved"
    else:
        return "needs_improvement"
```

---

## 6. Building the Graph with Loops

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(TweetState)

# Add nodes
graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

# Edges
graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")

# Conditional edge from evaluate
graph.add_conditional_edges(
    "evaluate",
    route_evaluation,
    {
        "approved": END,
        "needs_improvement": "optimize"
    }
)

# Loop edge: optimize → evaluate
graph.add_edge("optimize", "evaluate")

workflow = graph.compile()
```

---

## 7. Running the Workflow

```python
initial_state = {
    "topic": "Indian Railways",
    "tweet": "",
    "evaluation": "",
    "feedback": "",
    "iteration": 1,
    "max_iterations": 5,
    "tweet_history": [],
    "feedback_history": []
}

result = workflow.invoke(initial_state)

print("Final approved tweet:", result["tweet"])
print("Iterations used:", result["iteration"])
print("All tweets generated:", result["tweet_history"])
print("All feedback received:", result["feedback_history"])
```

---

## 8. Understanding the Loop

- The **conditional edge** from `evaluate` decides whether to exit (`END`) or go to `optimize`.
- The **normal edge** from `optimize` back to `evaluate` completes the loop.
- Each time `optimize` runs, it increments `iteration`. The router checks both approval status and iteration limit to avoid infinite loops.

---

## 9. Key Takeaways

- **Loops** in LangGraph are simply edges that go from a later node back to an earlier node.
- Use **conditional edges** to decide whether to loop again or exit.
- **Reducers** (`operator.add`) are essential for accumulating history across iterations – otherwise later values would overwrite earlier ones.
- Structured output ensures the evaluator always returns reliable fields.
- Always include a **max iteration limit** when building loops to prevent infinite runs.

---

## 10. Comparison with Other Workflow Types

| Workflow Type | Execution Pattern | Example |
|---------------|-------------------|---------|
| Sequential | Linear chain (A→B→C) | BMI calculator |
| Parallel | Multiple nodes run simultaneously | Essay evaluation (clarity, depth, language) |
| Conditional | Choose one path based on condition | Quadratic equation solver (case discriminant) |
| **Iterative (Looping)** | Repeat a cycle until condition met | Tweet generator (generate → evaluate → improve → re‑evaluate) |

---

## 010. How to build a Chatbot using LangGraph (36:37)

## Building a Chatbot with LangGraph – State, Memory & Persistence

This tutorial is the **first part** of a multi‑video series building a **production‑grade chatbot** using LangGraph.  
It covers:

1. **Design of a simple chatbot** (single‑node sequential workflow).
2. **State definition** – storing conversation history as a list of messages with a special reducer (`add_messages`).
3. **Building and compiling** the graph.
4. **Implementing a multi‑turn conversation loop** (while loop with user input).
5. **The core problem** – without persistence, each invocation starts from scratch → no memory of past messages.
6. **The solution** – using a **checkpointer** (`MemorySaver`) to persist state across invocations.
7. **Threading concept** – separating different conversation sessions via `thread_id`.
8. **How it works** – state is saved after each graph run and restored for the next run.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Chatbot as a workflow** | A simple sequential graph with a single node that calls an LLM. |
| 2 | **State design** | Must store the whole conversation history – a list of `BaseMessage` objects. |
| 3 | **Special reducer `add_messages`** | Built‑in LangGraph reducer that appends new messages to the list instead of replacing them. Essential for preserving history. |
| 4 | **Graph structure** | `START → chat_node → END` |
| 5 | **Multi‑turn loop** | A `while True` loop that collects user input, invokes the graph with the latest message, prints the AI response, and exits on keywords like `exit`, `quit`, `bye`. |
| 6 | **The memory problem** | Each `invoke()` creates a fresh state unless persistence is enabled. The loop calls `invoke()` repeatedly, so previous messages are lost. |
| 7 | **Persistence with checkpoints** | A **checkpointer** saves the state after each graph execution. The next invocation can resume from that saved state. |
| 8 | **`MemorySaver`** | An in‑memory checkpointer (stores state in RAM). Suitable for development; production uses database‑based checkpoints. |
| 9 | **`thread_id`** | A unique identifier for a conversation session. Different users (or different chats) have different `thread_id`. |
| 10 | **`config`** | A dictionary passed to `invoke()` containing `{"configurable": {"thread_id": "some_id"}}`. Tells LangGraph which persisted state to load. |
| 11 | **How persistence works** | After each graph run, the checkpointer saves the final state keyed by `thread_id`. On next invoke with the same `thread_id`, LangGraph loads that state before execution. |

---

## 1. Simple Chatbot Design

The chatbot is a **sequential workflow** with **only one node**:

```
START → chat_node → END
```

- The user sends a message.
- The `chat_node` calls an LLM with the entire conversation history.
- The LLM generates a response.
- The response is added to the history.

**Key insight:** The conversation history must be stored in the **state** and passed to the LLM each time.

---

## 2. State Definition with `add_messages` Reducer

We define a `ChatState` with a single field `messages`.  
We use `Annotated` with LangGraph’s built‑in `add_messages` reducer – it **appends** new messages to the list instead of replacing them.

```python
from typing import Annotated, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
```

**Why `add_messages`?**  
Without a reducer, assigning a new list would replace the old history. `add_messages` merges the new messages into the existing list, preserving the full conversation.

---

## 3. Building the Graph

### 3.1 Import required components

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
```

### 3.2 Define the LLM and the chat node

```python
llm = ChatOpenAI(model="gpt-4o-mini")

def chat_node(state: ChatState) -> dict:
    # The state already contains all previous messages.
    # Invoke the LLM with the whole message list.
    response = llm.invoke(state["messages"])
    # Return the new message as a list – `add_messages` will append it.
    return {"messages": [response]}
```

### 3.3 Build the graph

```python
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
```

### 3.4 Compile (without persistence – will lose memory)

```python
chatbot = graph.compile()
```

---

## 4. Multi‑Turn Conversation Loop (No Persistence – Broken)

This loop asks the user for input, invokes the graph, prints the AI’s response, and repeats until the user types `exit`/`quit`/`bye`.

```python
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    # Invoke the graph with the new message only (old history is lost!)
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    final_state = chatbot.invoke(initial_state)
    
    # Extract the last AI message
    ai_message = final_state["messages"][-1]
    print(f"AI: {ai_message.content}")
```

**The problem:** Each `invoke()` starts with a **fresh state** containing only the current user message. Previous messages are **not** included. Therefore, the LLM has no memory of earlier turns → it cannot answer “What’s my name?” after being told earlier.

---

## 5. The Solution: Persistence (Checkpoints)

LangGraph allows you to add a **checkpointer** that saves the state after every superstep.  
When you invoke the graph again with the same `thread_id`, it **loads the saved state** automatically.

### 5.1 Import `MemorySaver`

```python
from langgraph.checkpoint.memory import MemorySaver
```

### 5.2 Create a checkpointer and compile the graph with it

```python
checkpointer = MemorySaver()   # stores state in RAM
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)
```

### 5.3 Define a `thread_id` and a `config`

A **thread** represents one conversation session (e.g., one user, one chat). All invocations with the same `thread_id` share the persisted state.

```python
thread_id = "user_nitish"
config = {"configurable": {"thread_id": thread_id}}
```

### 5.4 Modified loop – pass `config` to each invoke

Now we pass the **same config** every time. The first invoke starts with an empty state; subsequent invokes load the previous state.

```python
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    # Prepare input – only the new user message; old history comes from checkpoint
    input_state = {"messages": [HumanMessage(content=user_input)]}
    final_state = chatbot.invoke(input_state, config=config)
    
    ai_message = final_state["messages"][-1]
    print(f"AI: {ai_message.content}")
```

**Now the bot remembers!**  
- First turn: user says “My name is Nitish”. Bot responds “Hello Nitish”.  
- Second turn: user asks “What is my name?”. The state loaded from checkpoint contains the previous messages, so the bot can answer correctly.

---

## 6. How Persistence Works Under the Hood

| Step | What happens |
|------|--------------|
| 1 | First `invoke()` with `config` – no saved state for this `thread_id`. Initial state is used. |
| 2 | Graph runs: `chat_node` reads `state["messages"]`, calls LLM, returns updated state. |
| 3 | After execution, the checkpointer **saves the final state** in memory (RAM) using `thread_id` as the key. |
| 4 | Second `invoke()` with same `config` – checkpointer loads the saved state, adds the new user message (via `add_messages`), then runs the graph. |
| 5 | The new final state is saved again, overwriting/updating the previous one. |
| 6 | This continues for every turn. |

Because we used `add_messages`, messages are appended, not replaced, so the complete conversation history accumulates.

---

## 7. Inspecting the Saved State

You can retrieve the full state for a thread at any time:

```python
saved_state = chatbot.get_state(config)
print(saved_state.values["messages"])
```

This prints all messages exchanged so far.

---

## 8. Limitations of `MemorySaver`

- **In‑memory only** – if the Python process restarts, all saved states are lost.
- **Not suitable for production** – real chatbots use a database checkpointer (e.g., `PostgresSaver`, `RedisSaver`) to persist state across server restarts.

The next video will cover:
- Database‑based persistence.
- Checkpointers in depth.
- Fault tolerance (resuming after crash).
- Human‑in‑the‑loop (interrupting and resuming).

---

## 9. Complete Working Code Example (With Persistence)

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, TypedDict
from langchain_core.messages import HumanMessage, BaseMessage

# 1. State definition with add_messages reducer
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 2. LLM and node function
llm = ChatOpenAI(model="gpt-4o-mini")

def chat_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 3. Build graph with checkpointer
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

# 4. Conversation loop with persistence
thread_id = "my_chat_session"
config = {"configurable": {"thread_id": thread_id}}

print("Chatbot ready. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break

    input_state = {"messages": [HumanMessage(content=user_input)]}
    final_state = chatbot.invoke(input_state, config=config)
    ai_message = final_state["messages"][-1]
    print(f"AI: {ai_message.content}")

# Optional: inspect full history
print("\n--- Full conversation history ---")
for msg in chatbot.get_state(config).values["messages"]:
    print(f"{msg.type}: {msg.content}")
```

---

## 10. Key Takeaways

- A chatbot is just a **sequential workflow** with one node that calls an LLM.
- The **state** must hold the entire conversation history.
- Use **`add_messages`** as a reducer to append new messages instead of overwriting.
- Without persistence, each `invoke()` starts fresh – the bot has **no memory**.
- **Checkpointers** save the state after each graph run.
- **`thread_id`** in the `config` tells LangGraph which saved state to load.
- **`MemorySaver`** stores state in RAM (development only). Production needs a database.
- This pattern is the foundation for more advanced features: human‑in‑the‑loop, fault tolerance, and multi‑user sessions.

---

## 011. Persistence in LangGraph (58:13)

## Persistence in LangGraph – Checkpoints, Threads, and State Recovery

This part of tutorial explains **persistence** – one of the most important foundation concepts in LangGraph. Persistence allows you to **save and restore the state of a workflow over time**, including intermediate states, not just the final result. This enables **fault tolerance** (resuming after crashes) and **chat history retention** (resuming past conversations).

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Persistence** | The ability to save and restore the state of a workflow over time. |
| 2 | **Without persistence** | After a graph finishes execution (`invoke` completes), the state is erased from memory. You cannot recover it. |
| 3 | **What persistence saves** | Not just the final state, but **every intermediate state** – at each checkpoint. |
| 4 | **Checkpoint** | A saved snapshot of the state at a particular moment during graph execution. |
| 5 | **Checkpointer** | The component in LangGraph that creates checkpoints and stores them (in memory or database). |
| 6 | **Superstep** | A round of execution that may contain one or more parallel steps. Each superstep becomes a checkpoint. |
| 7 | **Thread** | A unique identifier (`thread_id`) assigned to a specific execution of a workflow. All checkpoints for that execution are stored under that `thread_id`. |
| 8 | **Fault tolerance** | If a workflow crashes, you can resume from the last checkpoint instead of starting over. |
| 9 | **Chat history** | By saving all messages under a `thread_id`, you can later resume a conversation or list past conversations. |
| 10 | **Storage options** | In‑memory (`MemorySaver`) for development; database (PostgreSQL, Redis) for production. |

---

## 1. What is Persistence? (Simple Definition)

> **Persistence in LangGraph refers to the ability to save and restore the state of a workflow over time.**

### Recap: How LangGraph Works Without Persistence

- You define a **graph** (nodes = tasks, edges = execution order).
- You define a **state** (dictionary holding important data).
- You call `graph.invoke(initial_state)` – the graph runs step by step, updating the state.
- When the graph reaches `END`, the execution finishes and the **state is discarded** (erased from memory).
- If you run the graph again, you start with a **fresh, empty state** – no memory of previous runs.

### What Persistence Adds

Persistence **saves the state at multiple points** during execution (at every **superstep** / checkpoint). Later, you can:
- **Resume** a workflow from where it crashed (fault tolerance).
- **Restore** a previous conversation (chat history).
- **Inspect** intermediate states for debugging.

---

## 2. How Persistence Works – Checkpoints and Supersteps

LangGraph implements persistence using **checkpointers**. A checkpointer divides the graph execution into **checkpoints** (one per superstep) and saves the state at each checkpoint.

### What is a Superstep?

A **superstep** is a round of execution that may contain **one or more parallel steps**.  
Example graph:

```
START → Node1
          ↓
    ┌─────┴─────┐
    ↓     ↓     ↓
  Node2  Node3  Node4   (parallel, same superstep)
    ↓     ↓     ↓
    └─────┴─────┘
          ↓
         END
```

**Supersteps in this graph:**
- **Superstep 1:** `START → Node1`
- **Superstep 2:** `Node1 → (Node2, Node3, Node4)` – all three run in parallel, but together they form **one superstep**.
- **Superstep 3:** `(Node2, Node3, Node4) → END`

Each superstep produces a **checkpoint** – a saved snapshot of the entire state.

### Example: Intermediate State Saving

Suppose your state has a key `numbers` (list of integers) with a reducer that **appends** new values.

| Checkpoint | State (`numbers`) | What happened |
|------------|-------------------|----------------|
| 1 (after START) | `[1]` | Initial state |
| 2 (after Node1) | `[1, 2]` | Node1 added `2` |
| 3 (after parallel nodes) | `[1, 2, 3, 4, 5]` | Node2 added `3`, Node3 added `4`, Node4 added `5` |
| 4 (after END) | `[1, 2, 3, 4, 5]` | Final state (same as previous) |

All four snapshots are saved in the database (or memory). You can later restore any of them.

---

## 3. Threads – Isolating Different Executions

When you run the same graph multiple times (different initial states, or different users), you need a way to **separate the checkpoints** of each run. This is done with **threads**.

### What is a Thread?

A **thread** is a **unique identifier** (`thread_id`) that you assign to a specific execution of a workflow. All checkpoints generated during that execution are stored under that `thread_id`.

**Example:**

- **Thread 1** – initial state `numbers = [1]` → checkpoints saved under `thread_id = "user1_session1"`.
- **Thread 2** – initial state `numbers = [6]` → checkpoints saved under `thread_id = "user1_session2"`.
- **Thread 3** – another user, initial state `numbers = [1]` → saved under `thread_id = "user2_session1"`.

Without threads, checkpoints from different executions would mix together – you wouldn’t know which state belongs to which conversation or run.

### Using Threads for Chatbots

- When a user starts a new chat, create a new `thread_id`.
- Store every message exchange in the state under that `thread_id`.
- Later, when the user wants to resume a previous chat, pass the same `thread_id` – LangGraph will load the entire saved state.

---

## 4. Benefits of Persistence

| Benefit | Description |
|---------|-------------|
| **Fault tolerance** | If a workflow crashes (server down, API failure), you can resume from the last checkpoint instead of starting over. |
| **Chat history** | Users can resume past conversations; you can list all past conversations per user. |
| **Debugging** | Inspect intermediate states to see what happened at each step. |
| **Auditing** | Keep a record of every state change for compliance. |
| **Long‑running workflows** | Workflows that take days/weeks (e.g., hiring process) can be paused and resumed without losing progress. |

---

## 5. Code Example: Adding Persistence to a Graph

### Step 1: Import `MemorySaver` (or other checkpointer)

```python
from langgraph.checkpoint.memory import MemorySaver
```

### Step 2: Create a checkpointer instance

```python
checkpointer = MemorySaver()   # stores checkpoints in RAM (development)
```

For production, you would use a database checkpointer like `PostgresSaver` or `RedisSaver`.

### Step 3: Compile the graph with the checkpointer

```python
graph = StateGraph(MyState)
# ... add nodes and edges ...
workflow = graph.compile(checkpointer=checkpointer)
```

### Step 4: Define a `thread_id` and a `config` dictionary

```python
thread_id = "user_123_session_1"
config = {"configurable": {"thread_id": thread_id}}
```

### Step 5: Invoke the workflow with the config

```python
initial_state = {"messages": [HumanMessage(content="Hi, my name is Nitish")]}
final_state = workflow.invoke(initial_state, config=config)
```

### Step 6: Subsequent invocations with the same `thread_id`

Now the graph will **load the previous state** before executing.

```python
# Later in the same conversation
new_state = {"messages": [HumanMessage(content="What is my name?")]}
final_state2 = workflow.invoke(new_state, config=config)
# The AI will answer correctly because the state now contains the full history.
```

### Step 7: Retrieve the full state for a thread

```python
saved_state = workflow.get_state(config)
print(saved_state.values)   # all messages exchanged so far
```

### Step 8: List all checkpoints for a thread (optional)

```python
for checkpoint in workflow.list_checkpoints(config):
    print(checkpoint)
```

---

## 6. Complete Working Example: Chatbot with Persistence

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, TypedDict
from langchain_core.messages import HumanMessage, BaseMessage

# 1. State with add_messages reducer
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# 2. LLM and node
llm = ChatOpenAI(model="gpt-4o-mini")
def chat_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# 3. Graph with checkpointer
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

# 4. Conversation loop with persistence
thread_id = "nitish_conversation"
config = {"configurable": {"thread_id": thread_id}}

print("Chatbot ready. Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    input_state = {"messages": [HumanMessage(content=user_input)]}
    final_state = chatbot.invoke(input_state, config=config)
    print("AI:", final_state["messages"][-1].content)

# 5. Retrieve full history
full_state = chatbot.get_state(config)
print("\n--- Full conversation ---")
for msg in full_state.values["messages"]:
    print(f"{msg.type}: {msg.content}")
```

---

## 7. Key Takeaways

- **Persistence** = saving the state of a workflow so it can be restored later.
- **Checkpoints** are saved at every **superstep** (including intermediate states, not just final).
- **Checkpointer** is the component that saves/restores checkpoints (e.g., `MemorySaver`).
- **Threads** (`thread_id`) separate checkpoints from different executions (different users or different sessions).
- Without persistence, state is lost after `invoke()` finishes.
- Persistence enables **fault tolerance** (resume after crash) and **chat history** (resume old conversations).
- In production, use a **database checkpointer** (PostgreSQL, Redis) instead of `MemorySaver`.

---

## Persistence Implementation in LangGraph – Code, Checkpoints, Threads & Fault Tolerance (contd...)

This part of tutorial continues the persistence discussion with **practical coding examples**. It shows how to implement persistence using a **checkpointer** (`MemorySaver`), how to use **thread IDs** to separate different executions, and how to retrieve **final state** and **intermediate state histories**.  
The video also demonstrates two major benefits: **short‑term memory** (chat history) and **fault tolerance** (resuming after a crash).

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **`MemorySaver`** | A built‑in checkpointer that stores state in RAM (for development/demos). Production uses database checkpoints (PostgreSQL, Redis). |
| 2 | **Compiling with checkpointer** | Pass the checkpointer to `graph.compile(checkpointer=checkpointer)` – tells LangGraph to save state at every superstep. |
| 3 | **Thread ID** | A unique identifier for a workflow execution. All checkpoints are stored under this ID. |
| 4 | **Config dictionary** | `{"configurable": {"thread_id": "some_id"}}` – passed to `invoke()` to associate the run with a thread. |
| 5 | **`get_state(config)`** | Retrieves the **final state** of a thread after execution. |
| 6 | **`get_state_history(config)`** | Retrieves **all intermediate states** (checkpoints) for a thread – each checkpoint shows the state at a superstep boundary and the next node to execute. |
| 7 | **Fault tolerance** | If a workflow crashes, you can **resume** by calling `invoke(None, config=config)` – LangGraph will continue from the last checkpoint. |
| 8 | **Short‑term memory** | Persistence allows chatbots to remember conversation history across turns (by storing all messages under a thread ID). |
| 9 | **Resuming after crash** | You don’t need to restart from the beginning; the workflow continues exactly where it stopped. |

---

## 1. Example Workflow: Joke Generator with Explanation

**Workflow:**  
`START → generate_joke → generate_explanation → END`

- **Node 1 (`generate_joke`)**: Takes a topic (e.g., “pizza”), uses LLM to generate a joke.
- **Node 2 (`generate_explanation`)**: Takes the joke, generates an explanation.

**State definition:**

```python
from typing import TypedDict

class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str
```

---

## 2. Code with Persistence (Full Working Example)

```python
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

# 1. State
class JokeState(TypedDict):
    topic: str
    joke: str
    explanation: str

# 2. LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. Nodes
def generate_joke(state: JokeState) -> dict:
    prompt = f"Tell a short joke about {state['topic']}"
    response = llm.invoke(prompt)
    return {"joke": response.content}

def generate_explanation(state: JokeState) -> dict:
    prompt = f"Explain why this joke is funny: {state['joke']}"
    response = llm.invoke(prompt)
    return {"explanation": response.content}

# 4. Build graph
graph = StateGraph(JokeState)
graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_explanation", generate_explanation)
graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "generate_explanation")
graph.add_edge("generate_explanation", END)

# 5. Add persistence (MemorySaver checkpointer)
checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# 6. Run with a thread ID
thread_id = "1"   # unique for this execution
config = {"configurable": {"thread_id": thread_id}}

initial_state = {"topic": "pizza", "joke": "", "explanation": ""}
result = workflow.invoke(initial_state, config=config)
print(result["joke"])
print(result["explanation"])
```

---

## 3. Retrieving Final State and Intermediate States

### Get final state of a thread

```python
final_state = workflow.get_state(config)
print(final_state.values)   # {'topic': 'pizza', 'joke': '...', 'explanation': '...'}
```

### Get all intermediate states (checkpoints)

```python
for checkpoint in workflow.get_state_history(config):
    print(checkpoint.values)          # state at that checkpoint
    print(checkpoint.next)            # next node(s) to execute
```

**Output (conceptual):**

| Checkpoint | State | Next node |
|------------|-------|-----------|
| 1 (before START) | `{}` | `"START"` |
| 2 (before generate_joke) | `{"topic": "pizza"}` | `"generate_joke"` |
| 3 (before generate_explanation) | `{"topic": "pizza", "joke": "..."}` | `"generate_explanation"` |
| 4 (after END) | full state | `None` |

**Why 4 checkpoints?**  
Each superstep boundary creates a checkpoint. The graph has:
- Superstep 1: before any node
- Superstep 2: after `generate_joke`
- Superstep 3: after `generate_explanation` (but before END)
- Superstep 4: after END

---

## 4. Running Multiple Threads (Separate Conversations)

```python
# Thread 1: pizza
config1 = {"configurable": {"thread_id": "pizza_session"}}
workflow.invoke({"topic": "pizza", ...}, config=config1)

# Thread 2: pasta (different thread ID)
config2 = {"configurable": {"thread_id": "pasta_session"}}
workflow.invoke({"topic": "pasta", ...}, config=config2)

# Retrieve state for pizza session
pizza_state = workflow.get_state(config1).values
print(pizza_state["joke"])   # joke about pizza

# Retrieve state for pasta session
pasta_state = workflow.get_state(config2).values
print(pasta_state["joke"])   # joke about pasta
```

Checkpoints are stored **per thread** – they don’t mix.

---

## 5. Fault Tolerance – Resuming After a Crash

**Setup:** A 3‑step workflow where **Step 2 has a 30‑second delay**. We simulate a crash during the delay, then resume.

### Workflow State

```python
class FaultState(TypedDict):
    input: str
    step_one: str
    step_two: str
    step_three: str
```

### Node functions (with time delay in step two)

```python
import time

def step_one(state: FaultState) -> dict:
    print("Executing step one")
    return {"step_one": "done"}

def step_two(state: FaultState) -> dict:
    print("Executing step two (will take 30 seconds)")
    time.sleep(30)   # simulate long processing
    return {"step_two": "done"}

def step_three(state: FaultState) -> dict:
    print("Executing step three")
    return {"step_three": "done"}
```

### Build graph with checkpointer

```python
graph = StateGraph(FaultState)
graph.add_node("step_one", step_one)
graph.add_node("step_two", step_two)
graph.add_node("step_three", step_three)
graph.add_edge(START, "step_one")
graph.add_edge("step_one", "step_two")
graph.add_edge("step_two", "step_three")
graph.add_edge("step_three", END)

checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer)
```

### First run – crash during step two

```python
config = {"configurable": {"thread_id": "fault_demo"}}
initial_state = {"input": "start", "step_one": "", "step_two": "", "step_three": ""}

try:
    result = workflow.invoke(initial_state, config=config)
except KeyboardInterrupt:
    print("Simulated crash during step two")
```

**After crash, inspect the saved state:**

```python
state_before_crash = workflow.get_state(config)
print(state_before_crash.values)
# {'input': 'start', 'step_one': 'done', 'step_two': '', 'step_three': ''}
print(state_before_crash.next)   # ['step_two'] – the node that was interrupted
```

### Resume from the crash point

Simply call `invoke()` again with **the same config** and **`None` as the input** (meaning “resume where you left off”).

```python
# Resume – no initial state needed
resumed_state = workflow.invoke(None, config=config)
print(resumed_state)
# {'input': 'start', 'step_one': 'done', 'step_two': 'done', 'step_three': 'done'}
```

**How it works:**  
- When you pass `None` as the input, LangGraph loads the last checkpoint for that `thread_id` and continues execution from the `next` node(s) stored in the checkpoint.
- Step two runs again (from where it was interrupted), then step three runs.
- The workflow completes without re‑executing step one.

---

## 6. Benefits of Persistence – Summary Table

| Benefit | How Persistence Helps |
|---------|-----------------------|
| **Short‑term memory** (chatbots) | Store all messages under a thread ID. Later invocations with same ID load full history. |
| **Fault tolerance** | If the workflow crashes, you can resume from the last checkpoint (no need to restart from beginning). |
| **Human‑in‑the‑loop (HITL)** | You can pause a workflow, wait for human input, then resume from the same checkpoint (covered in later videos). |
| **Time travel** | You can go back to any previous checkpoint and replay from there (debugging, auditing). |

---

## 7. Key Takeaways for Code Implementation

- **`MemorySaver`** is a checkpoint that stores state in RAM – suitable only for demos. Production uses database checkpoints (e.g., `PostgresSaver`).
- **Always pass a `config` with `thread_id`** when using persistence. Without it, checkpoints cannot be distinguished.
- **`get_state(config)`** gives the **final state** of a thread.
- **`get_state_history(config)`** gives **every intermediate checkpoint** – each includes the state and the `next` node(s) to execute.
- To **resume after a crash**, call `invoke(None, config=config)` – LangGraph automatically loads the last checkpoint and continues.
- Different threads (different `thread_id`) have completely separate checkpoint histories – perfect for multi‑user chatbots.

---

## 8. What’s Next?

The next videos will build on persistence to implement:
- **Human‑in‑the‑loop** – pausing a workflow, waiting for human approval, then resuming.
- **Time travel** – going back to any previous checkpoint to re‑run or debug.
- **Production checkpoints** – using PostgreSQL or Redis for persistent storage across server restarts.

---

## Human‑in‑the‑Loop, Time Travel & Advanced Persistence (contd...)

This final part of the persistence tutorial explains **two advanced features** built on top of persistence:

1. **Human‑in‑the‑Loop (HITL)** – pausing a workflow to ask for human input (approval, decision), then resuming later.
2. **Time Travel** – replaying a workflow from any previous checkpoint, modifying state, and creating new branches for debugging or experimentation.

Both features rely entirely on **checkpointing** (persistence). The video also summarises the four benefits of persistence: short‑term memory, fault tolerance, HITL, and time travel.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Human‑in‑the‑Loop (HITL)** | A workflow is intentionally interrupted to wait for a human decision (e.g., approve a post). |
| 2 | **Why persistence is required** | Human input may take hours/days. The workflow must be **saved** (checkpointed) and later **resumed** from the exact point. |
| 3 | **Interrupt** | LangGraph can pause execution at a node and wait for external input. The state is saved in a checkpoint. |
| 4 | **Resume** | When human input arrives, call `invoke(None, config)` with the same thread ID – LangGraph loads the checkpoint and continues. |
| 5 | **Time Travel** | Going back to any previous checkpoint (using its `checkpoint_id`) and re‑executing the remaining nodes. |
| 6 | **Checkpoint ID** | Every checkpoint has a unique identifier. You can retrieve it from `get_state_history()`. |
| 7 | **Replay** | Calling `invoke(None, config, checkpoint_id=id)` re‑runs the workflow from that checkpoint. |
| 8 | **Update state** | You can modify the state at a checkpoint using `update_state(config, new_values, checkpoint_id)`. This creates a new branch in the execution history. |
| 9 | **Branching** | Each time you replay or update state, a new “branch” is created – the original execution remains untouched. |
| 10 | **Primary use cases** | HITL for approvals, time travel for debugging and experimentation. |

---

## 1. Human‑in‑the‑Loop (HITL)

### The Problem

Imagine a workflow that:
1. Generates a LinkedIn post from a topic.
2. Posts it to LinkedIn via API.

You want to **ask the human for approval** before posting. The human may respond in seconds, hours, or days.  
You cannot keep the workflow running in memory for that long (it would waste resources and risk crashes).

### The Solution: Persistence + Interrupt

- LangGraph **saves a checkpoint** right before the human approval step.
- It **interrupts** execution (stops without finishing).
- The state is persisted (in database or memory).
- Later, when human input arrives, you call `invoke(None, config=config)` – LangGraph loads the checkpoint and resumes exactly where it stopped.

**Conceptual diagram:**

```
START → generate_post → [HUMAN APPROVAL] → post_to_LinkedIn → END
                          ↑                     
                    (interrupt, save state)     
                    wait for human input...    
                    then resume                 
```

**Note:** The video does not show the actual code for HITL, but states that it will be covered in a dedicated video. The key takeaway is that **persistence is the foundation** – without it, HITL would be impossible for long delays.

---

## 2. Time Travel – Replaying and Branching

Time travel allows you to:
- Go back to any checkpoint (using its `checkpoint_id`).
- Re‑run the workflow from that point (replay).
- **Modify the state** at a checkpoint and then re‑run (creating a new branch).

### Example: Joke Generation Workflow

Workflow: `START → generate_joke → generate_explanation → END`

We already ran it with `topic = "pizza"`. Now we want to:
1. Go back to the checkpoint **before the joke was generated** (where only `topic` exists).
2. Replay from there (generating a new joke and explanation).

### Step 1: Get the checkpoint history

```python
for checkpoint in workflow.get_state_history(config):
    print(checkpoint.values, checkpoint.next)
```

**Output (conceptual):**

| Checkpoint ID | State | Next node |
|---------------|-------|-----------|
| `ckpt_1` | `{}` | `START` |
| `ckpt_2` | `{"topic": "pizza"}` | `"generate_joke"` |
| `ckpt_3` | `{"topic": "pizza", "joke": "..."}` | `"generate_explanation"` |
| `ckpt_4` | full state | `None` |

### Step 2: Replay from a specific checkpoint

```python
# Copy the checkpoint_id from history (e.g., ckpt_2)
replay_config = {"configurable": {"thread_id": "pizza_session", "checkpoint_id": "ckpt_2"}}
new_result = workflow.invoke(None, config=replay_config)
```

**What happens:** LangGraph loads the state `{"topic": "pizza"}` and re‑executes `generate_joke` and `generate_explanation`. The new joke may be different (LLM is probabilistic). The original execution is **preserved** – a new branch is created.

### Step 3: Update state at a checkpoint

You can also modify the state before replaying – e.g., change `topic` from `"pizza"` to `"samosa"`.

```python
from langgraph.checkpoint import Checkpoint

# Get the checkpoint_id where topic is "pizza"
old_checkpoint_id = "ckpt_2"

# Update the state
workflow.update_state(
    config={"configurable": {"thread_id": "pizza_session"}},
    values={"topic": "samosa"},
    checkpoint_id=old_checkpoint_id
)
```

Now a **new checkpoint** is created with the updated state. You can then replay from that new checkpoint.

```python
# Get the new checkpoint_id (from the branch)
new_checkpoint_id = ...  # fetch from history

replay_config = {"configurable": {"thread_id": "pizza_session", "checkpoint_id": new_checkpoint_id}}
new_result = workflow.invoke(None, config=replay_config)
```

**Result:** A joke about samosa is generated, and the original pizza joke remains untouched. You have created a **branch** in the execution history.

---

## 3. Code Example: Time Travel with Checkpoint IDs

This example assumes you have the joke workflow from the previous video with persistence.

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

# Assume JokeState, nodes, graph built with checkpointer
checkpointer = MemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# First run: pizza topic
config_pizza = {"configurable": {"thread_id": "test_session"}}
initial = {"topic": "pizza", "joke": "", "explanation": ""}
workflow.invoke(initial, config=config_pizza)

# Get all checkpoints
history = list(workflow.get_state_history(config_pizza))
# Find the checkpoint where topic is set but joke is not yet generated
target_checkpoint = None
for ckpt in history:
    if "topic" in ckpt.values and "joke" not in ckpt.values:
        target_checkpoint = ckpt
        break

if target_checkpoint:
    checkpoint_id = target_checkpoint.config["configurable"]["checkpoint_id"]
    print(f"Replaying from checkpoint {checkpoint_id}")
    
    # Replay without changing state
    replay_config = {"configurable": {"thread_id": "test_session", "checkpoint_id": checkpoint_id}}
    new_result = workflow.invoke(None, config=replay_config)
    print("New joke:", new_result["joke"])
```

**To update state before replay:**

```python
# Update topic from pizza to samosa at that checkpoint
workflow.update_state(
    config_pizza,
    {"topic": "samosa"},
    checkpoint_id=checkpoint_id
)

# Get the new checkpoint created after update
new_history = list(workflow.get_state_history(config_pizza))
updated_checkpoint = new_history[0]  # most recent
updated_checkpoint_id = updated_checkpoint.config["configurable"]["checkpoint_id"]

# Replay from the updated checkpoint
replay_config2 = {"configurable": {"thread_id": "test_session", "checkpoint_id": updated_checkpoint_id}}
samosa_result = workflow.invoke(None, config=replay_config2)
print("Samosa joke:", samosa_result["joke"])
```

---

## 4. Four Benefits of Persistence – Summary

| Benefit | Description | Use Case |
|---------|-------------|----------|
| **Short‑term memory** | Store conversation history per thread | Chatbots that remember past messages |
| **Fault tolerance** | Resume workflow after crash | Long‑running workflows (e.g., hiring process) |
| **Human‑in‑the‑Loop** | Pause for human input, resume later | Approvals, reviews, risk‑sensitive actions |
| **Time travel** | Replay from any checkpoint, modify state, branch | Debugging, experimentation, “what‑if” scenarios |

All four are made possible because LangGraph saves **every intermediate state** (at each superstep/checkpoint) and allows you to later **restore** any of those states and continue execution.

---

## 5. Key Takeaways

- **Human‑in‑the‑Loop** is implemented by **interrupting** the workflow at a specific node, saving the checkpoint, and later **resuming** with `invoke(None, config)`.
- **Time travel** uses `checkpoint_id` to go back to any saved state.
- You can **replay** from a checkpoint (`invoke(None, config_with_checkpoint_id)`).
- You can **update** the state at a checkpoint (`update_state()`) – this creates a **new branch** in the execution history.
- Every time you replay or update, a new checkpoint is added; the original history is preserved.
- These features are **advanced debugging tools** – not needed for simple workflows, but invaluable for complex agentic systems.
- Persistence (checkpointing) is the **foundation** for all of them. Without checkpoints, HITL and time travel would be impossible.

---

## 012. Building a Chatbot with UI in LangGraph & Streamlit (32:27)

This part of tutorial shows how to build a **user interface (UI)** for the LangGraph chatbot. The UI is built with **Streamlit**, a Python library for rapid web app development. The chatbot retains **conversation memory** (short‑term memory) across turns, and the UI displays messages in a clean chat format.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Two‑component architecture** | Separate the chatbot into **backend** (LangGraph graph + checkpointer) and **frontend** (Streamlit UI). |
| 2 | **Backend code** | Exactly the same LangGraph code from the previous video – defines state, graph, nodes, and checkpointer (`MemorySaver`). |
| 3 | **Frontend code** | Uses Streamlit's `st.chat_message` (for displaying messages) and `st.chat_input` (for user input). |
| 4 | **Streamlit behaviour** | The script re‑runs from top to bottom on every user interaction. Without persistence, conversation history is lost. |
| 5 | **`st.session_state`** | A dictionary that **persists across script re‑runs** – used to store the message history list. |
| 6 | **Message history structure** | A list of dictionaries, each with `role` (`"user"` or `"assistant"`) and `content` (message text). |
| 7 | **Integration with LangGraph** | Import the compiled `chatbot` object from the backend. Use `chatbot.invoke()` with a `config` containing `thread_id` and a state containing the new user message. |
| 8 | **Thread ID** | Required because the backend uses a checkpointer. Each conversation (or user) gets a unique `thread_id`. |
| 9 | **Full workflow** | User types → message appended to session state history → displayed → sent to LangGraph (with full history) → AI response appended → displayed. |
| 10 | **Code changes** | Only a small part of the frontend needs to be changed – replace the dummy “echo” response with a real LangGraph call. |

---

## 1. Architecture Overview

The chatbot is split into two Python files:

- **`langraph_backend.py`** – contains the LangGraph workflow (state, graph, nodes, checkpointer).  
  This is the same code as the previous video – a simple single‑node graph that calls an LLM with the full message history.

- **`streamlit_frontend.py`** – contains the Streamlit UI code. It imports the compiled `chatbot` from the backend and uses it to generate responses.

**Flow:**
```
User types message in Streamlit UI
         ↓
Streamlit script runs (top to bottom)
         ↓
Message added to session_state history
         ↓
History displayed (all previous messages)
         ↓
(If new user input) → call chatbot.invoke() with config + state
         ↓
AI response extracted from returned state
         ↓
AI response added to session_state history and displayed
```

---

## 2. Streamlit Basics – Chat Message & Chat Input

Streamlit provides two key components for chat interfaces:

### `st.chat_message(role)`

Creates a chat bubble for a specific role (`"user"` or `"assistant"`). Inside it, you put the message content (e.g., with `st.text()` or `st.write()`).

```python
import streamlit as st

# Display a user message
with st.chat_message("user"):
    st.text("Hi, how are you?")

# Display an assistant message
with st.chat_message("assistant"):
    st.text("I'm fine, thank you!")
```

### `st.chat_input(placeholder)`

Creates an input box at the bottom of the screen where the user can type. It returns the typed text when the user presses Enter.

```python
user_input = st.chat_input("Type your message here...")
```

**Important:** When the user presses Enter, the **entire Streamlit script re‑runs**. So any variables defined outside `session_state` are reset.

---

## 3. The Problem – Losing History on Re‑run

In a simple implementation without session state:

```python
user_input = st.chat_input("Say something")
if user_input:
    with st.chat_message("user"):
        st.text(user_input)
    with st.chat_message("assistant"):
        st.text(user_input)   # dummy echo
```

Each time you send a new message, the script re‑runs and **overwrites** the previous messages – only the latest message is shown.  
**Why?** Because the variable `user_input` is re‑created each time, and there is no storage for past messages.

---

## 4. Solution – Using `st.session_state` to Store History

`st.session_state` is a dictionary that **persists across script re‑runs**. You can store the entire conversation history as a list of messages.

**Structure of the history list:**
```python
[
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "What is my name?"},
    {"role": "assistant", "content": "You are Nitish."}
]
```

**Initialisation:**
```python
if "messages" not in st.session_state:
    st.session_state.messages = []   # empty list
```

**Displaying the history:**
```python
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])
```

**Adding a new message:**
```python
st.session_state.messages.append({"role": "user", "content": user_input})
```

When the script re‑runs, the loop over `st.session_state.messages` prints all past messages, so history is preserved.

---

## 5. Integrating LangGraph

The backend (`langraph_backend.py`) defines a compiled graph called `chatbot`. This graph uses a checkpointer (`MemorySaver`) and requires a `thread_id` in the `config`.

**Backend code (simplified):**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

llm = ChatOpenAI(model="gpt-4o-mini")

def chat_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)
```

Now, in the frontend, we import `chatbot`:

```python
from langraph_backend import chatbot
from langchain_core.messages import HumanMessage

# Define config with a thread_id
CONFIG = {"configurable": {"thread_id": "user123"}}

# When user sends a message:
if user_input:
    # Add user message to session history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Prepare state for LangGraph
    input_state = {"messages": [HumanMessage(content=user_input)]}
    
    # Invoke the graph with the config
    response_state = chatbot.invoke(input_state, config=CONFIG)
    
    # Extract the AI's reply (last message)
    ai_message = response_state["messages"][-1]
    ai_content = ai_message.content
    
    # Add AI message to session history
    st.session_state.messages.append({"role": "assistant", "content": ai_content})
```

Then the display loop will show both the user and assistant messages.

---

## 6. Final Working Code (Frontend)

```python
import streamlit as st
from langraph_backend import chatbot
from langchain_core.messages import HumanMessage

# 1. Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Config for persistence (thread_id)
CONFIG = {"configurable": {"thread_id": "user123"}}

# 3. Display all past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])

# 4. Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Call LangGraph
    input_state = {"messages": [HumanMessage(content=user_input)]}
    response_state = chatbot.invoke(input_state, config=CONFIG)
    ai_message = response_state["messages"][-1]
    ai_content = ai_message.content
    
    # Add AI message to history
    st.session_state.messages.append({"role": "assistant", "content": ai_content})
    
    # Re-run the script to display the new messages
    st.rerun()
```

---

## 7. Key Takeaways

| Step | What happens |
|------|--------------|
| 1 | User opens the page – `st.session_state.messages` is empty, no messages shown. |
| 2 | User types a message and presses Enter. |
| 3 | The script re‑runs. `user_input` is set to the typed text. |
| 4 | The user message is appended to `st.session_state.messages`. |
| 5 | LangGraph is called with the new user message (and the full conversation history is automatically included because the checkpointer loads previous messages based on `thread_id`). |
| 6 | The AI response is appended to `st.session_state.messages`. |
| 7 | The display loop runs and shows all messages (including the new ones). |
| 8 | `st.rerun()` forces a fresh re‑run to update the UI with the new messages. |

**Why does LangGraph remember previous messages?**  
Because the backend uses a checkpointer and we provide the same `thread_id` each time. The checkpointer loads the saved state (containing all previous messages) before executing the graph. The `add_messages` reducer appends the new user message, so the LLM receives the full history.

---

## 8. Summary – UI Development

- Use `st.chat_message` to display chat bubbles.
- Use `st.chat_input` to get user input.
- Store the conversation history in `st.session_state` as a list of `{"role": ..., "content": ...}` dictionaries.
- Every time the script runs, loop through `st.session_state.messages` and display them.
- When a new message arrives, append it to history, call LangGraph with the same `thread_id`, get the AI reply, append it, and `st.rerun()`.

The result is a fully functional chatbot with a clean UI and persistent short‑term memory, powered by LangGraph.

### Command to run streamlit UI - `streamlit run chatbot_streamlit_frontend.py`

---

## 013. Streaming in LangGraph (24:59)

This part of tutorial shows how to implement **streaming** – the ability to display LLM responses **token by token** (like a typewriter effect) – in the existing LangGraph chatbot with a Streamlit UI. Streaming dramatically improves user experience, especially for long outputs.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Streaming** | LLM sends tokens as soon as they are generated, instead of waiting for the full response. |
| 2 | **Problem without streaming** | For long responses (500-word blogs, code), the UI freezes for 5‑10 seconds, then the entire output appears at once – poor UX. |
| 3 | **Benefits of streaming** | Faster perceived response time, mimics human conversation, better readability for long/code outputs, allows early stopping (saves tokens/money), supports progress updates. |
| 4 | **Technical difference** | Use `graph.stream()` instead of `graph.invoke()`. `stream()` returns a **generator** that yields chunks as they arrive. |
| 5 | **Streamlit integration** | Use `st.write_stream(generator)` – it displays output with a typewriter effect and handles the UI automatically. |
| 6 | **Stream mode** | For simple text responses, use `stream_mode="messages"` – it yields message chunks. |
| 7 | **Backend unchanged** | The LangGraph graph itself (nodes, edges, checkpointer) remains unchanged. Only the invocation method changes. |
| 8 | **Frontend change** | Replace `chatbot.invoke()` with `chatbot.stream()` inside `st.write_stream()`. |

---

## 1. What is Streaming?

**Streaming** means the model starts sending tokens (words/characters) **as soon as they are generated**, instead of waiting for the entire response to be ready.

### Without Streaming (Blocking)

```
User: "Write a 500-word blog on cricket"
→ AI thinks for 8 seconds (screen blank)
→ Entire blog appears at once
→ User feels the app is frozen/broken
```

### With Streaming

```
User: "Write a 500-word blog on cricket"
→ "C" appears instantly
→ "Cr" appears
→ "Cri" appears
→ ... tokens appear one by one with a typewriter effect
→ User sees progress and stays engaged
```

---

## 2. Why Streaming is Important

| Benefit | Explanation |
|---------|-------------|
| **Faster perceived response** | User sees something happening immediately, even if total generation time is the same. |
| **Mimics human conversation** | Feels like someone is "typing" a response, building trust and engagement. |
| **Better for long outputs** | Code or long essays are easier to follow when they appear step by step. |
| **Early stopping** | If the response is going in the wrong direction, the user can stop it mid‑way, saving tokens (money). |
| **Progress updates** | For agentic workflows, you can show step‑by‑step progress (e.g., "Searching... → Found results → Generating..."). |

---

## 3. Technical Implementation in LangGraph

### The Core Change: `invoke()` → `stream()`

**Before (blocking):**
```python
result = chatbot.invoke(input_state, config=config)
ai_message = result["messages"][-1]
```

**After (streaming):**
```python
stream = chatbot.stream(input_state, config=config, stream_mode="messages")
for message_chunk, metadata in stream:
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)
```

### What is a Generator?

`chatbot.stream()` returns a **generator** – a special Python object that yields values **one at a time** on the fly, using the `yield` keyword instead of `return`.

```python
# Simple generator example
def count_up_to(n):
    for i in range(n):
        yield i   # returns i, then pauses until next iteration

gen = count_up_to(3)
next(gen)  # → 0
next(gen)  # → 1
next(gen)  # → 2
```

### How the Streaming Loop Works

```python
# Get the stream generator
stream = chatbot.stream(
    {"messages": [HumanMessage(content="Tell me a joke")]},
    config=config,
    stream_mode="messages"
)

# Iterate through each token chunk
for message_chunk, metadata in stream:
    # message_chunk contains the token (with content)
    # metadata contains additional info (e.g., node name)
    if message_chunk.content:
        # Print token without newline, flush immediately
        print(message_chunk.content, end="", flush=True)
```

**Key details:**
- `stream_mode="messages"` tells LangGraph to yield message chunks (tokens) as they are generated by the LLM.
- Each `message_chunk` is a piece of the final message.
- The loop continues until the LLM finishes generating.

---

## 4. Streamlit Integration: `st.write_stream()`

Streamlit provides a built‑in function `st.write_stream()` that handles the entire streaming UI.

### What `st.write_stream()` does

- Takes a **generator** that yields strings.
- Displays the strings one by one with a **typewriter effect**.
- Automatically updates the UI as new tokens arrive.

### Code Change in the Frontend

**Before (no streaming):**
```python
# Call invoke and wait for full response
response_state = chatbot.invoke(input_state, config=config)
ai_message = response_state["messages"][-1]
ai_content = ai_message.content

# Display as a complete message
with st.chat_message("assistant"):
    st.text(ai_content)
```

**After (with streaming):**
```python
# Get the stream generator
stream = chatbot.stream(input_state, config=config, stream_mode="messages")

# Create a generator that yields only the content
def token_generator():
    for message_chunk, metadata in stream:
        if message_chunk.content:
            yield message_chunk.content

# Display with typewriter effect
with st.chat_message("assistant"):
    response = st.write_stream(token_generator())

# Store the complete response in session state
st.session_state.messages.append({"role": "assistant", "content": response})
```

**Important:** `st.write_stream()` returns the **complete concatenated string** after the stream finishes, so you can store it in `session_state` for history.

---

## 5. Complete Working Example (Frontend with Streaming)

```python
import streamlit as st
from langraph_backend import chatbot
from langchain_core.messages import HumanMessage

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []

CONFIG = {"configurable": {"thread_id": "user123"}}

# Display all past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Prepare input for LangGraph
    input_state = {"messages": [HumanMessage(content=user_input)]}
    
    # Get the stream generator
    stream = chatbot.stream(input_state, config=CONFIG, stream_mode="messages")
    
    # Define a generator that yields only content
    def token_generator():
        for message_chunk, metadata in stream:
            if message_chunk.content:
                yield message_chunk.content
    
    # Display with typewriter effect and capture full response
    with st.chat_message("assistant"):
        full_response = st.write_stream(token_generator())
    
    # Store the full response in history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Re-run to update the UI
    st.rerun()
```

---

## 6. Architecture Overview

| Component | File | Changes |
|-----------|------|---------|
| **Backend** | `langraph_backend.py` | **No changes** – graph, nodes, checkpointer remain exactly the same. |
| **Frontend** | `streamlit_frontend.py` | **Replace `invoke()` with `stream()`** – use `st.write_stream()` for display. |

The backend still uses `graph.compile(checkpointer=checkpointer)` and `chatbot.invoke()` is still available – but we now use `chatbot.stream()` for a better UX.

---

## 7. Key Takeaways

- **Streaming is a UI/UX feature** – the backend (LLM) already supports token‑by‑token generation; streaming is just about displaying those tokens as they arrive.
- **`invoke()` returns the final result** – it blocks until the entire response is ready.
- **`stream()` returns a generator** – it yields tokens as they are generated, allowing real‑time display.
- **`st.write_stream()`** is the simplest way to implement streaming in Streamlit – it takes a generator and handles the typewriter effect.
- **Session state is still needed** – after streaming finishes, you need to store the complete response in `session_state` so it persists across re‑runs.
- **Backend remains unchanged** – the same LangGraph graph and checkpointer work perfectly with both `invoke()` and `stream()`.

### Useful Link
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 014. How to build a Resume Chat feature like ChatGPT? (39:39)

## Adding Resume Chat Feature to LangGraph Chatbot with Streamlit

This part of tutorial adds the **"Resume Chat"** feature – the ability to start **multiple conversations** and switch between them, just like ChatGPT. The UI gains a sidebar with:
- A **“New Chat”** button to start a fresh conversation.
- A list of **all past conversations** (by thread ID) that the user can click to resume.

The implementation uses **dynamic thread IDs** (UUIDs), stores all thread IDs in `st.session_state`, and retrieves conversation history from the LangGraph checkpointer using `get_state()`.

**No backend changes** are required – the existing LangGraph graph and `MemorySaver` checkpointer are sufficient.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Resume Chat** | Users can start new conversations and switch between existing ones; each conversation is identified by a unique `thread_id`. |
| 2 | **Dynamic thread IDs** | Use `uuid.uuid4()` to generate a unique ID for each new conversation – manual hard‑coded IDs won’t work. |
| 3 | **Session state** | Store the current `thread_id`, the message history list, and a list of all thread IDs (`chat_threads`) in `st.session_state`. |
| 4 | **Storing all thread IDs** | When a new conversation starts, append its thread ID to the `chat_threads` list so it appears in the sidebar. |
| 5 | **Loading a conversation** | Use `chatbot.get_state(config)` with the target thread ID to retrieve the saved state. Extract `messages` from the state. |
| 6 | **Message format conversion** | LangGraph returns `BaseMessage` objects (with `type` and `content`). Convert them to `{"role": "user" or "assistant", "content": ...}` dictionaries for the UI. |
| 7 | **UI components** | `st.sidebar.title()`, `st.sidebar.button()`, `st.sidebar.header()`, and loop over `chat_threads` to create clickable buttons. |
| 8 | **“New Chat” button logic** | Generate a new thread ID, save it to session, reset `messages` history, and append the new ID to `chat_threads`. |
| 9 | **Thread button click** | On click, update session state with that thread ID, load its messages, and set `messages` history accordingly. |
| 10 | **Ordering** | Newest conversations should appear at the top – reverse the `chat_threads` list when displaying. |
| 11 | **Homework** | Replace raw thread IDs with user‑friendly names (e.g., first few words of the first user message). |

---

## 1. Recap: Existing Chatbot Structure

- **Backend** (`langraph_backend.py`): LangGraph graph with `MemorySaver` checkpointer. No changes needed.
- **Frontend** (`streamlit_frontend.py`): Streamlit UI with `st.chat_message`, `st.chat_input`, and streaming.

The backend stores all conversation states (messages) in RAM (since we use `MemorySaver`). Each conversation is keyed by `thread_id`.

---

## 2. Step‑by‑Step Implementation

### 2.1 Add Sidebar UI

We add a sidebar with a title, a “New Chat” button, and a “My Conversations” header. Later, we’ll populate the list of threads.

```python
# Sidebar UI
with st.sidebar:
    st.title("LangGraph Chat Bot")
    if st.button("New Chat"):
        # will be implemented later
        pass
    st.header("My Conversations")
    # will display thread list later
```

### 2.2 Utility Function: Generate Thread ID

```python
import uuid

def generate_thread_id():
    return str(uuid.uuid4())
```

### 2.3 Initialize Session State

We need three session state keys:
- `thread_id`: the current conversation’s thread ID.
- `messages`: list of message dicts for the current conversation.
- `chat_threads`: list of all thread IDs ever created.

```python
# Inside the main script (top)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []
```

### 2.4 Utility: Add Thread to List

To avoid duplicates, we check if the thread ID is already present before appending.

```python
def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)
```

When the app first loads, we add the initial thread ID:

```python
add_thread(st.session_state.thread_id)
```

### 2.5 “New Chat” Logic

When the user clicks “New Chat”:
- Generate a new thread ID.
- Replace the current `thread_id` in session.
- Clear `messages` history.
- Add the new thread ID to `chat_threads`.

```python
def reset_chat():
    new_thread_id = generate_thread_id()
    st.session_state.thread_id = new_thread_id
    st.session_state.messages = []  # clear messages
    add_thread(new_thread_id)       # add to list
```

Then in the sidebar button:

```python
if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()
```

### 2.6 Display Threads in Sidebar

We loop through `chat_threads` (reversed for newest first) and create a button for each thread ID.

```python
for thread_id in reversed(st.session_state.chat_threads):
    if st.sidebar.button(thread_id, key=thread_id):
        # load this conversation
        pass
```

### 2.7 Load Conversation from a Thread

When a thread button is clicked, we need to:
1. Set the current `thread_id` to the clicked one.
2. Retrieve the saved state from LangGraph using `chatbot.get_state()`.
3. Extract the messages from the state.
4. Convert messages from `BaseMessage` objects to the UI’s dictionary format.
5. Update `messages` in session state.
6. Rerun to refresh UI.

**Utility function to load messages:**

```python
def load_conversation(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state_snapshot = chatbot.get_state(config)
    if state_snapshot is None:
        return []
    messages = state_snapshot.values.get("messages", [])
    # Convert BaseMessage to dict
    converted = []
    for msg in messages:
        if msg.type == "human":
            role = "user"
        else:
            role = "assistant"
        converted.append({"role": role, "content": msg.content})
    return converted
```

**Button handler:**

```python
if st.sidebar.button(thread_id, key=thread_id):
    st.session_state.thread_id = thread_id
    st.session_state.messages = load_conversation(thread_id)
    st.rerun()
```

### 2.8 Display Messages (Existing Code)

The existing code already iterates over `st.session_state.messages` and displays them. No change needed.

### 2.9 Streaming with Current Thread

In the main chat input handler, we already use `chatbot.stream()` with the current `thread_id` from session state. That part remains unchanged.

---

## 3. Full Frontend Code (Final)

Below is the complete `streamlit_frontend.py` with all features.

```python
import streamlit as st
import uuid
from langraph_backend import chatbot
from langchain_core.messages import HumanMessage

# ------------------------------
# Utility functions
# ------------------------------
def generate_thread_id():
    return str(uuid.uuid4())

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

def reset_chat():
    new_thread_id = generate_thread_id()
    st.session_state.thread_id = new_thread_id
    st.session_state.messages = []
    add_thread(new_thread_id)

def load_conversation(thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    if state is None:
        return []
    messages = state.values.get("messages", [])
    converted = []
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        converted.append({"role": role, "content": msg.content})
    return converted

# ------------------------------
# Session state initialisation
# ------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = []

# Ensure the current thread is in the list
add_thread(st.session_state.thread_id)

# ------------------------------
# Sidebar UI
# ------------------------------
with st.sidebar:
    st.title("LangGraph Chat Bot")
    if st.button("New Chat"):
        reset_chat()
        st.rerun()
    st.header("My Conversations")
    for thread_id in reversed(st.session_state.chat_threads):
        if st.button(thread_id, key=thread_id):
            st.session_state.thread_id = thread_id
            st.session_state.messages = load_conversation(thread_id)
            st.rerun()

# ------------------------------
# Main chat area
# ------------------------------
# Display all messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.text(msg["content"])

# Chat input
user_input = st.chat_input("Type your message...")
if user_input:
    # Add user message to UI history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Prepare state for LangGraph
    input_state = {"messages": [HumanMessage(content=user_input)]}
    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    # Stream AI response
    stream = chatbot.stream(input_state, config=config, stream_mode="messages")
    
    def token_generator():
        for msg_chunk, metadata in stream:
            if msg_chunk.content:
                yield msg_chunk.content

    with st.chat_message("assistant"):
        full_response = st.write_stream(token_generator())

    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.rerun()
```

---

## 4. Display Logical Names Instead of Thread IDs

**Problem:** Showing raw UUIDs like `8a7f9c4d-...` is ugly and not user‑friendly.

**Solution:** Store a **title** for each conversation (e.g., first 30 characters of the first user message, or “New Chat” if empty). We can store a dictionary mapping `thread_id` → `title` in session state.

### Implementation Steps

1. In session state, add a dictionary `thread_titles`.
2. When a new conversation starts (either at app load or “New Chat”), set its title.
3. In the sidebar, display the title instead of the thread ID.
4. When loading a conversation, we need to also set the title (optional).

We can set the title when the user sends their **first message** – we can update the title for the current thread. Or we can set it when the thread is created.

**Simpler approach:** At thread creation, set title as “New Chat”. Then, when the user sends the first message, update the title to the first 30 characters of that message.

Here’s the modified code:

```python
# Add to session state initialisation
if "thread_titles" not in st.session_state:
    st.session_state.thread_titles = {}

def set_thread_title(thread_id, title):
    st.session_state.thread_titles[thread_id] = title

def get_thread_title(thread_id):
    return st.session_state.thread_titles.get(thread_id, "New Chat")
```

When a new thread is created (in `reset_chat()` or initialisation), set its title to `"New Chat"`.

```python
def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)
        # Set default title
        st.session_state.thread_titles[thread_id] = "New Chat"
```

When the user sends a message, if the current thread has title `"New Chat"` (or empty), update the title to the first 30 characters of the user input.

```python
if user_input:
    # ... before invoking LangGraph ...
    current_title = get_thread_title(st.session_state.thread_id)
    if current_title == "New Chat" or not current_title:
        new_title = user_input[:30] + ("..." if len(user_input) > 30 else "")
        set_thread_title(st.session_state.thread_id, new_title)
    # ... rest of code ...
```

Then in the sidebar, display the title:

```python
for thread_id in reversed(st.session_state.chat_threads):
    title = get_thread_title(thread_id)
    if st.sidebar.button(title, key=thread_id):
        # load conversation
        ...
```

Now the sidebar shows meaningful names like “Write a 500-word blog…” instead of UUIDs.

---

## 5. Key Takeaways

- **Resume Chat** is implemented purely on the frontend – the backend remains unchanged.
- **Thread IDs** must be generated dynamically (UUID) so each conversation is unique.
- **Session state** is crucial for persisting the list of threads and current conversation between Streamlit reruns.
- **Loading a conversation** uses `chatbot.get_state()` to retrieve the saved state from the checkpointer.
- **Message format conversion** is necessary because LangGraph uses `BaseMessage` objects, while the UI expects simple dicts with `role` and `content`.
- **Adding logical names** to threads greatly improves UX; store a title per thread in session state.

---

## 6. Next Steps

As a next step, the `MemorySaver` (RAM‑based checkpointer) will be replaced with a **database‑based checkpointer** (e.g., PostgreSQL), so conversations survive server restarts. This will make the “Resume Chat” feature **persistent across app reloads**.

---

## 015. LangGraph + SQLite | Chatbot with Database Integration (28:47)

## 🎯 Main Problem Being Solved

**Issue:** Earlier, the chatbot used `InMemorySaver` (RAM-based storage). This meant all conversations were lost when:
- The application closed
- The page was refreshed
- The server restarted

**Solution:** Replace `InMemorySaver` with `SQLiteSaver` (database-based persistence). Now conversations remain intact even after application restarts.

---

## 📊 Before vs After Comparison

| Feature | Before (InMemorySaver) | After (SQLiteSaver) |
|---------|------------------------|---------------------|
| Storage Location | RAM | SQLite Database (.db file) |
| Persistence | ❌ Lost on restart | ✅ Saved permanently |
| Multi-thread Support | ❌ | ✅ Works with all threads |
| Data Recovery | ❌ Cannot recover old chats | ✅ Can resume any old conversation |

---

## 🔧 Backend Changes (LangGraph Code)

### 1. Install Required Library

```bash
pip install langgraph-checkpoint-sqlite
```

### 2. Import the New Checkpointer

**Old Import:**
```python
from langgraph.checkpoint.memory import MemorySaver
```

**New Import:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver
```

### 3. Setup SQLite Database

```python
import sqlite3

# Create database connection
conn = sqlite3.connect("chatbot.db", check_same_thread=False)

# Create checkpoint using SQLite
checkpointer = SqliteSaver(conn)
```

**Key Point:** `check_same_thread=False` is important because we'll be using multiple threads for different conversations. SQLite normally restricts to single thread, but this parameter bypasses that restriction.

### 4. Full Backend Code Structure

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
# ... other imports ...

# Database setup
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

# Create graph with checkpointer
graph = builder.compile(checkpointer=checkpointer)

# Function to retrieve all existing threads
def retrieve_all_threads():
    all_threads = set()
    for checkpoint_tuple in checkpointer.list(None):
        config = checkpoint_tuple.config
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            all_threads.add(thread_id)
    return list(all_threads)
```

---

## 📝 Important Concepts Explained

### What is a Checkpointer?

A checkpointer saves the state of your conversation at different points. Think of it like a save point in a video game - you can always come back to where you left off.

### Why SQLite?

| Database Type | Use Case | Pros | Cons |
|---------------|----------|------|------|
| InMemorySaver | Testing | Fast, no setup | Data lost on restart |
| SQLiteSaver | Prototyping | Persistent, easy setup | Not for production scale |
| PostgreSQL | Production | Scalable, robust | Complex setup |

### What are Threads?

Threads represent different conversation sessions. Example:
- **Thread 1:** "My name is Nitish" conversation
- **Thread 2:** "My name is Rahul" conversation  
- **Thread 3:** "Recipe for Biryani" conversation

Each thread has its own separate chat history.

---

## 🗄️ Understanding Checkpoint Storage

### How Checkpoints Work

When you run a conversation, multiple checkpoints are created:

```
Thread 1 → [Checkpoint 1] → [Checkpoint 2] → [Checkpoint 3]
                     ↓               ↓               ↓
                Start of      After user     After AI
                conversation   message        reply
```

### Visualizing Database

You can view the SQLite database using VS Code extensions:
1. Search for "SQLite Viewer" extension
2. Install "SQLite Viewer" by Florian Klampfer
3. Click on `chatbot.db` file to see all checkpoints

**What you'll see:**
- Multiple checkpoint entries per thread
- Each checkpoint stores: Thread ID, Messages, State
- You can click on checkpoints to see actual messages

---

## 🎨 Frontend Changes (Streamlit Code)

### The Key Change

**Before (No Persistence):**
```python
# Initialize with empty list
chat_threads = session_state.get("chat_threads", [])
```

**After (With Persistence):**
```python
from langgraph_database_backend import chat_bot, retrieve_all_threads

# Initialize with existing threads from database
chat_threads = session_state.get("chat_threads", retrieve_all_threads())
```

### Complete Frontend Code Structure

```python
import streamlit as st
from langgraph_database_backend import chat_bot, retrieve_all_threads

# Initialize session state
if "messages_history" not in st.session_state:
    st.session_state.messages_history = {}
    
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None
    
if "chat_threads" not in st.session_state:
    # THIS IS THE KEY CHANGE - Load from database
    st.session_state.chat_threads = retrieve_all_threads()

# Rest of the UI code remains same...
```

---

## 🧪 Testing the Implementation

### Test 1: Persistent Memory

**Step 1:** Run the code with message
```python
response = chat_bot.invoke(
    {"messages": [("user", "Hi, my name is Nitish")]},
    config={"configurable": {"thread_id": "thread_1"}}
)
```

**Step 2:** Close application, reopen

**Step 3:** Ask "What is my name?"
```python
response = chat_bot.invoke(
    {"messages": [("user", "What is my name?")]},
    config={"configurable": {"thread_id": "thread_1"}}
)
# Output: "Your name is Nitish"
```

**✅ Success:** The bot remembers even after restart!

### Test 2: Multiple Threads

**Thread 1:**
```python
# Create conversation in thread 1
chat_bot.invoke(
    {"messages": [("user", "My name is Nitish")]},
    config={"configurable": {"thread_id": "thread_1"}}
)
```

**Thread 2:**
```python
# Create conversation in thread 2
chat_bot.invoke(
    {"messages": [("user", "My name is Rahul")]},
    config={"configurable": {"thread_id": "thread_2"}}
)
```

**Result:** Both threads are stored separately and can be retrieved independently.

---

## 🔍 Checking Database Content

### Code to Check Threads

```python
# Get all unique thread IDs
def retrieve_all_threads():
    all_threads = set()
    for checkpoint_tuple in checkpointer.list(None):
        config = checkpoint_tuple.config
        thread_id = config.get("configurable", {}).get("thread_id")
        if thread_id:
            all_threads.add(thread_id)
    return list(all_threads)

# Usage
existing_threads = retrieve_all_threads()
print(existing_threads)  # Output: ['thread_1', 'thread_2']
```

### What's in Each Checkpoint?

Each checkpoint contains:
- **config**: Thread ID and other settings
- **values**: Messages in the conversation
- **metadata**: Timestamps, next node, etc.

---

## 📝 Complete Implementation Checklist

### Backend Steps:
- [ ] Install `langgraph-checkpoint-sqlite`
- [ ] Import `SqliteSaver` instead of `MemorySaver`
- [ ] Create SQLite connection with `check_same_thread=False`
- [ ] Connect checkpointer to database
- [ ] Create `retrieve_all_threads()` function
- [ ] Update graph compilation with new checkpointer

### Frontend Steps:
- [ ] Import `retrieve_all_threads()` from backend
- [ ] Replace empty list initialization with `retrieve_all_threads()`
- [ ] Ensure session state handles thread management
- [ ] Test persistence by restarting application

### Testing Steps:
- [ ] Test single conversation persistence
- [ ] Test multiple thread isolation
- [ ] Close and reopen application
- [ ] Verify old conversations are accessible
- [ ] Create new conversations alongside old ones

---

## 🎯 Key Takeaways

1. **Persistence Solution:** SQLite checkpointer replaces in-memory storage
2. **Data Safety:** All conversations survive application restarts
3. **Thread Management:** Multiple independent conversation threads possible
4. **User Experience:** Users can return to any old conversation anytime
5. **Visualization:** SQLite Viewer extension helps understand data structure

---

## 🚀 Common Errors & Solutions

### Error 1: "SQLite objects created in a thread can only be used in that same thread"
**Solution:** Add `check_same_thread=False` when creating connection
```python
conn = sqlite3.connect("chatbot.db", check_same_thread=False)
```

### Error 2: Module not found
**Solution:** Install the library
```bash
pip install langgraph-checkpoint-sqlite
```

### Error 3: No existing threads showing
**Solution:** Ensure database file (`chatbot.db`) exists and checkpointer is properly connected

---

## 💡 Real-World Example Flow

```
1. User opens app
   ↓
2. App loads existing threads from database
   (Thread 1: Nitish, Thread 2: Rahul)
   ↓
3. User clicks Thread 1
   ↓
4. App retrieves all messages from that thread
   ("My name is Nitish", "What is my name?")
   ↓
5. User asks new question
   ↓
6. New message is added to database
   ↓
7. User closes app
   ↓
8. User opens app next day
   ↓
9. All threads and messages are still there! ✅
```

---

## 016. LangSmith Crash Course | Observability in GenAI (02:07:39)

## LangSmith: Observability & Evaluation for LLM Applications

This part of tutorial introduces **LangSmith** – a unified platform for debugging, testing, and monitoring LLM-based applications. It explains why observability is critical, what LangSmith traces, core concepts, and provides hands‑on code demos with LangChain and LangGraph.

---

## 📌 Why Observability Matters – Three Real‑World Scenarios

### 1. Latency Spike in a Cover Letter Generator
- A startup built an app that takes a job description (JD), fetches a student’s portfolio from Google Drive, matches skills, and generates a tailored cover letter.
- Normally, the whole process takes ~2 minutes.
- Suddenly, it starts taking 7–10 minutes for some users.
- The team **cannot pinpoint** which component (JD reading, portfolio scanning, matching, LLM calls, proofreading) is causing the delay.
- **Problem**: No visibility into internal step‑by‑step performance.

### 2. Cost Surge in a Research Assistant
- An agentic research assistant fetches academic papers, extracts key points, summarizes them, and allows Q&A.
- Usual cost per report: ₹0.50.
- Suddenly, some reports cost ₹2.00, while others remain at ₹0.50.
- **Root cause** (hypothetical): A prompt update said *“keep refining until the report is perfect”*, causing the agent to loop multiple times for certain topics.
- **Problem**: Without tracing, you cannot see how many iterations or which steps consume extra tokens.

### 3. Hallucinations in a RAG Chatbot (TCS HR Bot)
- A RAG‑based chatbot answers employee queries (leave policy, notice period, insurance) by retrieving company documents.
- Users start getting **wrong answers** (e.g., “you can take leave whenever you want”).
- Hallucinations can come from:
  - The **retriever** (wrong documents fetched).
  - The **generator** (poor prompt, weak model, or not enforcing context‑grounded answers).
- **Problem**: You can’t see what the retriever returned, what the final prompt looked like, or the LLM’s raw output – making debugging nearly impossible.

> **Common thread**: LLM systems are **non‑deterministic, complex, and black‑box** – errors don’t produce clear stack traces. You need **observability** to see inside.

---

## 🔍 What is Observability?

> **Observability** is the ability to understand a system’s internal state by examining its external outputs (logs, metrics, traces). It allows you to diagnose issues, understand performance, and improve reliability by analysing data generated by the system.

In practice, it means:
- Seeing **every step** of your LLM workflow.
- Knowing what **input** each component received and what **output** it produced.
- Measuring **latency**, **token usage**, and **cost** per component.
- Tracking **errors** and adding **custom tags/metadata** for filtering.

---

## 🧰 LangSmith – Unified Observability & Evaluation Platform

**LangSmith** is a platform that provides **out‑of‑the‑box observability** for LLM applications built with LangChain, LangGraph, or even custom code.

**What LangSmith traces (at a granular level):**
- **Inputs & outputs** of every run (user query, LLM prompt, retrieved context, final response).
- **All intermediate steps** (prompt templates, LLM calls, parsers, retriever outputs, tool calls).
- **Latency** (total time and per‑component time).
- **Token usage** (input tokens, output tokens) and **cost** (based on model pricing).
- **Errors** (exceptions, failed API calls).
- **Metadata** (model name, temperature, version, user‑defined tags).
- **Custom tags & metadata** – you can attach your own labels (e.g., “production”, “experiment”, “user_id”).

---

## 📚 Core Concepts: Projects, Traces, Runs

- **Project** – The entire application you are building (e.g., “Cover Letter Generator”, “Research Assistant”).
- **Trace** – One single execution of your application (one user query, one request).
- **Run** – Each individual component step within a trace (e.g., a prompt template, an LLM call, a parser, a tool invocation).

> **Analogy**: A **Project** is a notebook. A **Trace** is one page of writing. A **Run** is one word on that page.

---

## ⚙️ Setup & Installation

1. **Clone the repository** (contains all demo scripts).
2. **Create a virtual environment** and install dependencies.
3. **Get a LangSmith API key** from [smith.langchain.com](https://smith.langchain.com) (Settings → API Keys → Create).
4. **Create a `.env` file** with the following variables:

```env
OPENAI_API_KEY=your-openai-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_PROJECT=langsmith_demo   # default project name
```

After this, **any** LangChain/LangGraph code you run will automatically send traces to LangSmith – no code changes required!

---

## 💻 Code Demo 1: Simple LLM Call

### Code (from `simple_llm_call.py`)

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Define a simple chain
prompt = PromptTemplate.from_template("What is the capital of {country}?")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()
chain = prompt | model | parser

# Invoke with a query
result = chain.invoke({"country": "Peru"})
print(result)  # Lima
```

### What appears in LangSmith?

- A new trace appears under the project `langsmith_demo`.
- The trace shows:
  - **Input**: `{"country": "Peru"}`
  - **Output**: `"Lima"`
  - **Runs**: three runs – `PromptTemplate`, `ChatOpenAI`, `StrOutputParser`.
- Each run shows its own input/output and timing.
- The LLM run also shows **token usage** and **cost**.

---

## 🧪 Code Demo 2: Sequential Chain with Custom Project, Tags & Metadata

This example creates a **two‑step workflow**: generate a detailed report on a topic, then summarise it into 5 bullet points.

### Key customizations shown:

- **Set project name in code** (override `.env`).
- **Specify different models** for each step.
- **Add custom tags & metadata** to the trace.
- **Set a custom run name**.

### Code (from `sequential_chain.py`)

```python
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Override project name from .env
os.environ["LANGCHAIN_PROJECT"] = "sequential_llm_app"

# Two different models
model1 = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
model2 = ChatOpenAI(model="gpt-4o", temperature=0.5)

# Step 1: generate report
prompt1 = PromptTemplate.from_template(
    "Generate a detailed report on {topic}."
)
chain1 = prompt1 | model1 | StrOutputParser()

# Step 2: summarise
prompt2 = PromptTemplate.from_template(
    "From the following text, generate 5 bullet points summary:\n\n{report}"
)
chain2 = prompt2 | model2 | StrOutputParser()

# Combine into a sequential chain
full_chain = chain1 | chain2

# Add custom tags, metadata, and run name via config
config = {
    "tags": ["LLM App", "Report Generation", "Summarisation"],
    "metadata": {
        "model1": "gpt-4o-mini",
        "model1_temperature": 0.7,
        "model2": "gpt-4o",
        "model2_temperature": 0.5,
        "parser": "StrOutputParser"
    },
    "run_name": "sequential_chain"
}

result = full_chain.invoke({"topic": "Unemployment in India"}, config=config)
print(result)
```

### What you see in LangSmith:

- A new project `sequential_llm_app` is created.
- A trace with the custom name `sequential_chain`.
- The trace shows 6 runs (prompt1, model1, parser, prompt2, model2, parser) in order.
- **Tags** appear on the trace (e.g., “LLM App”, “Report Generation”).
- **Metadata** appears on the trace level, showing model details.
- **Each run** has its own latency, token usage, and cost.
- The model runs automatically show the model name and temperature as metadata.

---

## 🧩 Code Demo 3: Retrieval-Augmented Generation (RAG) Trace

*(This is mentioned but not fully coded in the transcript – but the pattern is the same.)*

A RAG pipeline with:
- Retriever (vector store query)
- Prompt with context
- LLM call
- Output parser

LangSmith will show the retriever’s input (query), output (retrieved documents), the final prompt (combining query + context), and the LLM’s response. This helps debug hallucinations by checking if the context was relevant and if the prompt enforced grounding.

---

## ✨ Additional Features Shown

- **Custom run name** – `run_name` in `config` changes the trace name from auto‑generated (“RunnableSequence”) to something meaningful.
- **Per‑run tagging** – LangSmith automatically tags steps in a sequence with step numbers (e.g., “Sequence: step 1”, “Sequence: step 2”).
- **Project selection** – choose project either in `.env` or override in code with `os.environ["LANGCHAIN_PROJECT"]`.

---

## 📊 Benefits Summary

| Benefit | How LangSmith helps |
|---------|----------------------|
| **Latency debugging** | See which component takes the most time. |
| **Cost tracking** | See token usage and cost per run and per component. |
| **Error analysis** | If an API call fails, the error is captured. |
| **Hallucination diagnosis** | Inspect retrieved documents, prompt, and LLM output side‑by‑side. |
| **A/B testing** | Compare traces with different models/prompts using tags/metadata. |
| **Auditing** | Keep a full history of all requests and responses. |
| **User feedback** | Attach user scores/feedback to traces (advanced). |

---

## ✅ Key Takeaways

- **LangSmith is not a replacement** for your code – it’s a **drop‑in observability layer**.
- **Zero‑code change** for basic tracing – just set environment variables.
- **Custom tags/metadata** allow you to filter and group traces for analysis.
- **Granular run‑level details** show exactly what each component did.
- **Essential for production** LLM apps – without it, debugging is a nightmare.

---

## Tracing RAG Applications with LangSmith

This part of tutorial explains **why RAG applications are hard to debug** and how **LangSmith** provides the observability needed to diagnose issues. It covers two major problems with RAG in production – **retriever vs generator errors** and **latency** – and demonstrates practical solutions using LangSmith traces, custom `@traceable` decorators, and FAISS vector indexes with caching.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Retriever errors** | The retriever fetches irrelevant or wrong documents, leading to poor answers. |
| 2 | **Generator errors** | Even with good context, the LLM hallucinates or ignores the provided context. |
| 3 | **RAG debugging challenge** | Without tracing, you cannot tell if the problem is with the retriever or the generator. |
| 4 | **LangSmith tracing** | Traces every intermediate step – query, retrieved documents, final prompt, LLM response. |
| 5 | **Default tracing limitation** | LangSmith only traces LangChain runnables (where `invoke()` is called). Non-runnable steps (PDF loading, chunking) are not traced by default. |
| 6 | **`@traceable` decorator** | Allows you to trace **any** Python function, even if it doesn't use LangChain runnables. |
| 7 | **Latency problem** | In a naive RAG implementation, the PDF is loaded, chunked, and embedded on **every run**. |
| 8 | **Solution: persistent index** | Build a FAISS index once and store it locally. On subsequent runs, load the existing index instead of rebuilding. |
| 9 | **When index is rebuilt** | When PDF content changes, chunking parameters change, embedding model changes, or the index does not exist. |
| 10 | **Hierarchical tracing** | With `@traceable`, you can create nested traces – one parent trace contains the setup pipeline and the query pipeline together. |
| 11 | **Custom tags and metadata** | Attach tags and metadata to each component to enable filtering and search in LangSmith UI. |

---

## 1. Why RAG Apps Are Hard to Debug

A RAG (Retrieval-Augmented Generation) system:
1. User asks a question.
2. Retriever fetches relevant documents from a knowledge base.
3. LLM receives both the question and the retrieved context.
4. LLM generates an answer.

### Two Types of Errors

| Error Type | Description | Root Cause |
|------------|-------------|------------|
| **Retriever Error** | Irrelevant/wrong documents are fetched. | Bad retrieval algorithm, poor embeddings, wrong chunking, wrong query formulation. |
| **Generator Error** | LLM ignores the context or hallucinates. | Weak prompt (not enforcing "answer only from context"), poor model, or misunderstanding. |

**The core problem:** When the final answer is wrong, you cannot tell whether the retriever or the generator caused the failure – because you cannot see the intermediate steps.

---

## 2. How LangSmith Solves This

LangSmith traces **every intermediate step**:
- User query
- Documents retrieved by the retriever
- The final prompt (combining query + context)
- The LLM response
- Token usage and latency per component

**Result:** You can see exactly what the retriever returned and what the LLM received, making it easy to diagnose whether the problem is retrieval or generation.

---

## 3. Demo RAG Application

The demo uses a PDF book ("Introduction to Statistical Learning") and answers questions about it.

### Naive RAG Implementation (Version 1)

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import FAISS

# Load PDF
loader = PyPDFLoader("book.pdf")
docs = loader.load()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# Define prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer only from the provided context. If not found, say 'I don't know'."),
    ("user", "Context: {context}\nQuestion: {question}")
])

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Build chain with parallel retrieval
chain = (
    RunnableParallel({
        "context": retriever | lambda docs: "\n".join([d.page_content for d in docs]),
        "question": RunnablePassthrough()
    })
    | prompt
    | model
    | parser
)

# Query
result = chain.invoke("Who is the author of this book?")
print(result)
```

---

## 4. Problem 1 – Incomplete Tracing

When we run this code, LangSmith only traces the **chain part** (the runnables). The PDF loading, chunking, and embedding steps are **not traced** because they are not LangChain runnables.

**What you see in LangSmith:**
- `RunnableParallel` (question passthrough + retrieval)
- `ChatPromptTemplate`
- `ChatOpenAI`
- `StrOutputParser`

**What you don't see:**
- PDF loading time
- Chunking configuration
- Embedding model used
- Vector store creation time

---

## 5. Solution 1 – Using `@traceable` Decorator

LangSmith provides a `@traceable` decorator that can trace **any Python function**.

```python
from langsmith import traceable

@traceable(name="load_pdf")
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

@traceable(name="split_documents")
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vector_store")
def build_vector_store(docs, embedding_model="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever()

@traceable(name="setup_pipeline")
def setup_pipeline(file_path):
    docs = load_pdf(file_path)
    chunks = split_documents(docs)
    retriever = build_vector_store(chunks)
    return retriever
```

**Result:** Now LangSmith traces **the entire pipeline**, including the setup steps.

---

## 6. Problem 2 – Latency on Every Run

In the current implementation, **every run** loads the PDF, splits it, and generates embeddings. This is wasteful because:
- The PDF content never changes.
- The embeddings are the same each time.
- The setup takes ~15–20 seconds, while the actual query takes ~1–2 seconds.

**Ideal behaviour:**
- First run: build the vector index and store it.
- Subsequent runs: load the existing index and query it (much faster).

---

## 7. Solution 2 – Persistent FAISS Index with Caching

We build a **persistent index** that is saved locally and reused.

### Key logic:

```python
import os
from pathlib import Path

def build_index_if_not_exists(file_path, index_path=".indexes", 
                              chunk_size=1000, chunk_overlap=150,
                              embedding_model="text-embedding-3-small"):
    """
    Builds a FAISS index if it doesn't exist or if parameters have changed.
    Otherwise, loads the existing index.
    """
    # Check if index already exists
    index_dir = Path(index_path)
    index_file = index_dir / "faiss_index"
    
    if index_file.exists():
        # Check if parameters match (PDF file, chunk size, embedding model)
        # If everything matches, load the existing index
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vectorstore = FAISS.load_local(index_dir, embeddings)
        return vectorstore.as_retriever()
    
    # If not, build the index
    docs = load_pdf(file_path)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    # Save the index
    vectorstore.save_local(index_dir)
    return vectorstore.as_retriever()
```

### When the index is rebuilt:

| Condition | Trigger |
|-----------|---------|
| **First run** | No existing index → build. |
| **PDF changes** | Different file content, modified time, or size → rebuild. |
| **Chunk parameters change** | `chunk_size` or `chunk_overlap` changed → rebuild. |
| **Embedding model changes** | Different model name → rebuild. |

### Result:

| Run | Time | What happens |
|-----|------|--------------|
| **First run** | ~15–20 sec | Build index from scratch. |
| **Subsequent runs** | ~1–5 sec | Load existing index and query. |

---

## 8. Complete Flow with Tracing + Persistence

The final code combines:

- **`@traceable`** for full observability.
- **Persistent FAISS** for speed.
- **Custom tags and metadata** for filtering.

```python
import os
from pathlib import Path
from langsmith import traceable
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

os.environ["LANGCHAIN_PROJECT"] = "rag_chatbot"

# ---------- Setup functions with traceable ----------
@traceable(name="load_pdf", tags=["pdf", "loader"], metadata={"loader": "PyPDFLoader"})
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

@traceable(name="split_documents", tags=["chunking"])
def split_documents(docs, chunk_size=1000, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

@traceable(name="build_vector_store", tags=["embeddings", "vector_store"],
           metadata={"embedding_model": "text-embedding-3-small"})
def build_vector_store(docs, embedding_model="text-embedding-3-small"):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

@traceable(name="setup_pipeline")
def setup_pipeline(file_path, index_path=".indexes", chunk_size=1000, chunk_overlap=150):
    index_dir = Path(index_path)
    index_dir.mkdir(exist_ok=True)
    
    # Check if index exists (simplified logic)
    if (index_dir / "faiss_index").exists():
        print("Loading existing index...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = FAISS.load_local(index_dir, embeddings)
        return vectorstore.as_retriever()
    
    print("Building index from scratch...")
    docs = load_pdf(file_path)
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    vectorstore = build_vector_store(chunks)
    vectorstore.save_local(index_dir)
    return vectorstore.as_retriever()

# ---------- RAG Query ----------
@traceable(name="rag_query", run_type="chain")
def rag_query(question, retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer only from the provided context. If not found, say 'I don't know'."),
        ("user", "Context: {context}\nQuestion: {question}")
    ])
    model = ChatOpenAI(model="gpt-4o-mini")
    parser = StrOutputParser()
    
    chain = (
        RunnableParallel({
            "context": retriever | lambda docs: "\n".join([d.page_content for d in docs]),
            "question": RunnablePassthrough()
        })
        | prompt
        | model
        | parser
    )
    return chain.invoke(question)

# ---------- Main ----------
if __name__ == "__main__":
    retriever = setup_pipeline("book.pdf")
    answer = rag_query("Who is the author of this book?", retriever)
    print(answer)
```

**What LangSmith now shows:**

- One parent trace (`setup_pipeline`) containing:
  - `load_pdf` with tags `["pdf", "loader"]` and metadata `{"loader": "PyPDFLoader"}`
  - `split_documents` with tag `["chunking"]`
  - `build_vector_store` with tag `["embeddings", "vector_store"]` and metadata `{"embedding_model": "text-embedding-3-small"}`
- A separate trace for `rag_query` showing:
  - `RunnableParallel` (question pass-through + retrieval)
  - `ChatPromptTemplate`
  - `ChatOpenAI`
  - `StrOutputParser`

**On subsequent runs:**
- The `setup_pipeline` trace shows **`load_index`** (reusing the existing FAISS index) instead of rebuilding.
- Latency drops from ~15–20 seconds to ~1–2 seconds.

---

## 9. Advanced: Custom Tags and Metadata

```python
@traceable(
    name="load_pdf",
    tags=["pdf", "loader"],
    metadata={
        "loader": "PyPDFLoader",
        "file_name": "book.pdf",
        "pages": 441
    }
)
def load_pdf(file_path):
    ...
```

These tags and metadata appear in the LangSmith UI and are **searchable** – you can filter traces by model, loader, chunk size, or any custom metadata.

---

## 10. Key Takeaways

- **RAG is hard to debug** because errors can come from the retriever or the generator, and you can't see inside the black box.
- **LangSmith provides granular tracing** – every step, input, output, latency, token usage, and cost.
- **Default LangSmith tracing** only covers LangChain runnables (chains, models, parsers).
- **`@traceable` decorator** allows you to trace any Python function, even those without LangChain.
- **RAG latency can be reduced** by building a persistent FAISS index and reusing it.
- **Custom tags and metadata** make traces searchable – essential for large production systems.
- **Full end‑to‑end tracing** (including setup) helps you identify whether the issue is in the retriever, the prompt, or the LLM.

---

## 11. Summary of Problems and Solutions

| Problem | Solution |
|---------|----------|
| Only partial tracing (chain only) | Use `@traceable` on all functions |
| Can't tell retriever vs generator errors | Full trace shows both the retrieved docs and the final prompt |
| Slow on every run (PDF loading + embedding) | Persist FAISS index locally and reuse |
| Traces are in two separate pieces | Use `@traceable` with hierarchy – one parent trace containing both setup and query |

---

## Tracing Agentic AI Applications with LangSmith (Contd...)

This final section of the tutorial demonstrates how LangSmith traces **agentic applications** – specifically a **ReAct agent** that uses tools (DuckDuckGo search and a weather API) to answer complex queries. The tutorial shows step‑by‑step how the agent’s **Think → Act → Observe** loop is captured in LangSmith traces, making debugging and understanding agent behaviour much easier.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **ReAct agent** | An agent that follows a **Reason + Act** loop: Think (reason about the problem), Act (call a tool), Observe (get the result), and repeat until a final answer is reached. |
| 2 | **Scratchpad** | A working memory where the agent stores all intermediate thoughts, actions, and observations. |
| 3 | **LangSmith trace for agents** | Shows every iteration of the ReAct loop – each thought, tool call, and observation is a separate run within the trace. |
| 4 | **Why tracing agents is essential** | Agents are autonomous and can take multiple steps. Without tracing, you cannot see why they made certain decisions or if they are stuck in a loop. |
| 5 | **Tool call tracing** | Each tool call (search, weather API) is traced with its input, output, latency, and token usage. |
| 6 | **Scratchpad evolution** | LangSmith shows how the scratchpad grows with each iteration – new thoughts, actions, and observations are appended. |
| 7 | **Complex queries** | Agents can chain tools – e.g., search for a birth place, then use that result to get weather data. LangSmith shows the entire chain. |
| 8 | **Cost and latency monitoring** | LangSmith provides total token usage and cost per agent run, plus latency per step. |
| 9 | **Debugging agent failures** | If an agent gives a wrong answer, you can trace back to see which tool call or reasoning step went wrong. |
| 10 | **LangGraph integration** | Future videos will build agents with LangGraph – and LangSmith will be integrated to trace those as well. |

---

## 1. The ReAct Agent Demo

The agent has **two tools**:
1. **DuckDuckGo search** – for searching general information.
2. **Weather API** – for getting current temperature of a city.

### Simple Query: “What is the current temperature of Gurugram?”

The agent’s reasoning (visible in LangSmith trace):

| Step | Component | Content |
|------|-----------|---------|
| 1 | **Thought** | “I should use the get_weather_data tool.” |
| 2 | **Action** | `get_weather_data` with input `“Gurugram”` |
| 3 | **Observation** | Weather data: temperature 30°C, humidity 60%, wind speed 5 km/h |
| 4 | **Final answer** | “The current temperature of Gurugram is 30°C.” |

### Complex Query: “Identify the birthplace of Kalpana Chawla and give its current temperature.”

The agent must:
1. **Search** for Kalpana Chawla’s birthplace → finds Karnal.
2. **Get weather** for Karnal → returns temperature.
3. **Combine** both to give the final answer.

This requires **two tool calls** in sequence. LangSmith shows both steps clearly.

---

## 2. What LangSmith Traces in an Agent

### Trace Structure (for a single agent run)

```
RunnableSequence (top level)
├── Agent Scratchpad Initialization
├── Prompt Template (with system prompt + question)
├── ChatOpenAI (LLM call #1)
├── Tool Call: DuckDuckGo Search
│   ├── Input: "Kalpana Chawla birthplace city"
│   └── Output: search results (HTML/text)
├── Agent Scratchpad Update (append thought + action + observation)
├── Prompt Template (with updated scratchpad)
├── ChatOpenAI (LLM call #2 – reasoning about next step)
├── Tool Call: get_weather_data
│   ├── Input: "Karnal"
│   └── Output: weather JSON
├── Agent Scratchpad Update (append observation)
├── Prompt Template (final prompt with full history)
├── ChatOpenAI (LLM call #3 – generate final answer)
└── Final output
```

### Key Observations

- **Every LLM call** is traced – you can see the prompt sent and the response received.
- **Every tool call** is traced – you see the exact input and output (including raw API responses).
- **Scratchpad updates** are visible – you can see how the agent’s memory evolves.
- **Latency, token usage, and cost** are shown for each run and each sub‑component.

---

## 3. Why Tracing Agents is Even More Important

| Without LangSmith | With LangSmith |
|-------------------|----------------|
| You see only the final output. | You see every thought, action, and observation. |
| You cannot tell why the agent made a decision. | You can see the exact reasoning that led to each decision. |
| If the agent gives a wrong answer, you don’t know which step failed. | You can pinpoint the exact tool call or reasoning step that caused the error. |
| Cost spikes are hard to diagnose. | You can see which tool calls or repeated iterations are consuming tokens. |
| Loops (agent repeating the same step) are invisible. | Loops are visible as repeated LLM calls and tool calls in the trace. |

---

## 4. Code Example: Tracing a ReAct Agent

Below is a simplified version of the agent code used in the demo (with LangSmith tracing enabled automatically via environment variables).

```python
import os
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain.prompts import PromptTemplate

# LangSmith tracing is enabled by setting environment variables
# (already in .env file)
os.environ["LANGCHAIN_PROJECT"] = "react_agent_demo"

# ---------- Tools ----------
search_tool = DuckDuckGoSearchRun()
weather_tool = OpenWeatherMapAPIWrapper()  # wrapped as a tool
tools = [search_tool, weather_tool]

# ---------- Prompt ----------
prompt = PromptTemplate.from_template("""
You are a helpful assistant with access to the following tools:
{tools}

Use the following format:
Question: the input question
Thought: consider what to do
Action: the tool to use
Action Input: the input to the tool
Observation: the result from the tool
... (repeat Thought/Action/Observation as needed)
Thought: I now know the final answer
Final Answer: the answer to the question

Question: {input}
""")

# ---------- Agent ----------
llm = ChatOpenAI(model="gpt-4o-mini")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# ---------- Query ----------
result = agent_executor.invoke({
    "input": "Identify the birthplace of Kalpana Chawla and give its current temperature."
})
print(result["output"])
```

**What appears in LangSmith:**
- A single trace under the project `react_agent_demo`.
- Inside the trace, you see the iterative Thought → Action → Observation loop.
- Each tool call appears as a separate run with its own input/output and timing.

---

## 5. Comparing Tracing Types

| Application Type | Default LangSmith Tracing | With `@traceable` |
|------------------|---------------------------|-------------------|
| **Simple LLM call** | ✅ Full trace (prompt → model → parser) | Not needed |
| **RAG (chain only)** | ✅ Full trace of the chain | Setup steps (PDF loading, chunking, embedding) are not traced – need `@traceable` |
| **RAG (with persistent index)** | ✅ Full trace of query | Setup and index loading are traced with `@traceable` |
| **Agent** | ✅ Full trace of every iteration | Not needed – LangChain’s `AgentExecutor` is a runnable and is automatically traced |

---

## 6. Key Takeaways for Agent Tracing

- **Agents are complex** – they make multiple decisions and tool calls before returning an answer.
- **LangSmith shows the entire decision path** – each Thought, Action, and Observation is a separate run.
- **You can see why the agent chose a tool** – the LLM’s reasoning is captured in the prompt and response.
- **Tool inputs and outputs are visible** – you can verify that the agent gave the correct input to the tool and that the tool returned the expected data.
- **Debugging wrong answers** – if the agent returns a wrong answer, you can trace back to see if it was a search error, a weather API error, or a reasoning error.
- **Cost monitoring** – you can see how many LLM calls were made and how many tokens were used.

---

## 7. Final Conclusion – Why LangSmith Matters

| Problem | LangSmith Solution |
|---------|-------------------|
| **LLM applications are black boxes** | Full transparency – every component, every step, every input and output is visible. |
| **Errors don't produce stack traces** | You can inspect the exact prompt, context, and model response that caused the error. |
| **Costs can spike unexpectedly** | See token usage and cost per run and per component. |
| **Agent behaviour is unpredictable** | Trace the entire reasoning loop to understand why the agent made certain decisions. |
| **Debugging is manual and slow** | All traces are stored, searchable, and comparable. |

**Takeaway:** LangSmith is an essential tool for anyone building production‑grade LLM applications – especially agents, which are inherently complex and non‑deterministic.

---

## LangGraph + LangSmith Integration & Advanced LangSmith Features

This part of lecture of the LangSmith tutorial covers two major topics:

1. **How LangSmith integrates with LangGraph** – tracing graphs, nodes, and the entire workflow execution.
2. **Advanced LangSmith features** beyond observability: monitoring & alerting, evaluation, prompt experimentation, dataset creation, user feedback, and collaboration.

The instructor uses a **UPSC essay evaluation workflow** (built with LangGraph) to demonstrate how LangSmith traces parallel nodes, structured outputs, and the complete graph execution.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **LangGraph + LangSmith integration** | Since both products are from the same team, integration is seamless. |
| 2 | **Graph execution → Trace** | When you execute a LangGraph workflow, the entire execution becomes **one trace** in LangSmith. |
| 3 | **Node → Run** | Each node in the graph becomes **one run** inside the trace. |
| 4 | **Function-level tracing** | You can also trace the individual Python functions inside nodes using `@traceable`. |
| 5 | **Parallel nodes** | LangSmith correctly shows parallel execution – nodes run simultaneously and their outputs merge at the next node. |
| 6 | **Structured output nodes** | Nodes using `with_structured_output` show the schema and parsed output in the trace. |
| 7 | **Monitoring** | Aggregates metrics across multiple traces – latency, token usage, cost, error rates. |
| 8 | **Alerting** | Set alerts when metrics drift outside acceptable ranges (e.g., latency > 5 seconds). |
| 9 | **Evaluation** | Systematically measure LLM output quality using test datasets and metrics like faithfulness, relevance, completeness. |
| 10 | **Prompt Experimentation** | A/B test different prompt versions on the same dataset to find the best performer. |
| 11 | **Dataset creation** | Build and version datasets for evaluation and fine-tuning; support manual annotation. |
| 12 | **User feedback** | Capture thumbs up/down or structured feedback from users, linked to specific traces. |
| 13 | **Collaboration** | Share trace links, dashboards, and prompts with team members for effective debugging and iteration. |

---

## 1. LangGraph + LangSmith Integration – How It Works

### The Core Mapping

| LangGraph Concept | LangSmith Concept |
|-------------------|-------------------|
| **Entire Graph Execution** (one run of the workflow) | **One Trace** |
| **Each Node** (task/step in the graph) | **One Run** inside the trace |
| **Node's Python function** | Can be traced as a separate run using `@traceable` |

### Example: UPSC Essay Evaluation Graph

The graph structure:

```
START
  ├── evaluate_language (parallel)
  ├── evaluate_analysis (parallel)
  └── evaluate_clarity (parallel)
          ↓
    final_evaluation_node
          ↓
         END
```

- Three parallel nodes: language, analysis, clarity.
- Each node returns a **feedback text** and a **score**.
- The final node aggregates all feedbacks and calculates the average score.

**What LangSmith shows:**

- **One trace** for the entire graph execution.
- **Three runs** for the parallel nodes (they execute simultaneously).
- **One run** for the final evaluation node.
- **Additional runs** if `@traceable` is used on the node functions.

### Code Structure

```python
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import TypedDict, List

# ---------- State ----------
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: List[int]
    average_score: float

# ---------- Structured Output Schema ----------
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback")
    score: int = Field(description="Score out of 10", ge=0, le=10)

# ---------- LLM with Structured Output ----------
model = ChatOpenAI(model="gpt-4o-mini")
structured_model = model.with_structured_output(EvaluationSchema)

# ---------- Node Functions (with @traceable) ----------
@traceable(name="evaluate_language")
def evaluate_language(state: UPSCState) -> dict:
    prompt = f"Evaluate the language quality of this essay:\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {
        "language_feedback": result.feedback,
        "individual_scores": [result.score]   # reducer will merge
    }

@traceable(name="evaluate_analysis")
def evaluate_analysis(state: UPSCState) -> dict:
    prompt = f"Evaluate the depth of analysis in this essay:\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {
        "analysis_feedback": result.feedback,
        "individual_scores": [result.score]
    }

@traceable(name="evaluate_clarity")
def evaluate_clarity(state: UPSCState) -> dict:
    prompt = f"Evaluate the clarity of thought in this essay:\n{state['essay']}"
    result = structured_model.invoke(prompt)
    return {
        "clarity_feedback": result.feedback,
        "individual_scores": [result.score]
    }

@traceable(name="final_evaluation")
def final_evaluation(state: UPSCState) -> dict:
    # Generate overall feedback
    prompt = f"""
    Based on these feedbacks, create a single summarised feedback:
    Language: {state['language_feedback']}
    Analysis: {state['analysis_feedback']}
    Clarity: {state['clarity_feedback']}
    """
    overall = model.invoke(prompt)
    
    # Calculate average score
    avg = sum(state["individual_scores"]) / len(state["individual_scores"])
    
    return {
        "overall_feedback": overall.content,
        "average_score": round(avg, 2)
    }

# ---------- Build Graph ----------
graph = StateGraph(UPSCState)
graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_clarity", evaluate_clarity)
graph.add_node("final_evaluation", final_evaluation)

# Parallel edges from START to all three evaluators
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_clarity")

# All three go to final_evaluation
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_clarity", "final_evaluation")

graph.add_edge("final_evaluation", END)

# ---------- Compile and Execute ----------
workflow = graph.compile()

config = {
    "run_name": "evaluate_upsc_essay",
    "tags": ["essay_evaluation", "upsc"],
    "metadata": {"model": "gpt-4o-mini"}
}

initial_state = {
    "essay": "India has many smart students...",
    "language_feedback": "",
    "analysis_feedback": "",
    "clarity_feedback": "",
    "overall_feedback": "",
    "individual_scores": [],
    "average_score": 0.0
}

result = workflow.invoke(initial_state, config=config)
print(result)
```

### What Appears in LangSmith

| Trace Element | What You See |
|---------------|--------------|
| **Trace name** | `evaluate_upsc_essay` (custom run name) |
| **Trace tags** | `["essay_evaluation", "upsc"]` |
| **Trace metadata** | `{"model": "gpt-4o-mini"}` |
| **Runs (parallel)** | Three runs – `evaluate_language`, `evaluate_analysis`, `evaluate_clarity` – all with the same parent, showing they ran in parallel. |
| **Each run** | Shows the input state, the prompt, the structured output (feedback + score), latency, token usage, and cost. |
| **Final run** | `final_evaluation` run shows the aggregated feedback and average score. |
| **Function-level runs** | Each `@traceable` function appears as a separate run inside the node run. |

**Why this is powerful:**
- You can see **exactly** which node took the most time.
- You can see **what prompt** was sent to each LLM and **what response** was received.
- You can see **cost per node** – if one node uses more tokens than others, you can identify it immediately.
- Parallel execution is correctly visualised – three nodes running at the same time.

---

## 2. Advanced LangSmith Features

### 2.1 Monitoring & Alerting

**Monitoring** = tracking **multiple traces over time** to understand system health.

| Metric | What It Tracks |
|--------|----------------|
| **Trace count** | How many requests your app handles per day. |
| **Latency** | Average response time across all requests. |
| **Token usage** | Average input/output tokens per request. |
| **Cost** | Average cost per trace. |
| **Error rate** | Percentage of traces with errors. |
| **Tool usage** | How often tools are called. |

**Alerting** = notifying your team when metrics go outside acceptable ranges.

```python
# Conceptual example – you'd set this in LangSmith UI, not in code
alert_rule = {
    "project": "langgraph_essay_checker",
    "metric": "latency",
    "condition": "> 5.0 seconds",
    "action": "send_slack_notification"
}
```

**Why monitoring matters:**
> “In production, issues often appear first as patterns across multiple runs, rather than as a single trace. Monitoring helps you catch early signals before they impact users at scale.”

### 2.2 Evaluation

**Evaluation** = systematically measuring the quality of your LLM outputs against gold‑standard datasets or custom metrics.

**Common evaluation metrics:**
- **Faithfulness** – Does the answer stick to the provided context? (No hallucinations)
- **Relevance** – Is the answer relevant to the question?
- **Completeness** – Does the answer cover all aspects of the question?
- **Hallucination** – Does the model make up information?

**Method: LLM as a Judge**
- Use a **second LLM** to evaluate the output of your primary LLM.
- The judge LLM scores the output on various criteria.

```python
# Conceptual – using LangSmith's built-in evaluators
from langsmith.evaluation import evaluate

# Define a dataset of questions + expected answers
dataset = [
    {"question": "What is the capital of India?", "expected": "New Delhi"},
    {"question": "What is 2+2?", "expected": "4"}
]

# Run evaluation on your chain
evaluate(
    chain.invoke,
    data=dataset,
    evaluators=["hallucination", "faithfulness", "relevance"]
)
```

### 2.3 Prompt Experimentation

**Prompt experimentation** = systematically testing different prompt versions to find the best one.

**How it works:**
- Define a **test dataset** (questions + expected answers).
- Create **multiple prompt versions** (A, B, C…).
- Run the **same dataset** through each prompt version.
- Compare results using **evaluation metrics**.
- Choose the prompt that performs best.

**Example:**

| Prompt Version | Avg Score | Latency | Cost |
|----------------|-----------|---------|------|
| Prompt A (current) | 8.2/10 | 2.5s | $0.05 |
| Prompt B (new) | 8.7/10 | 2.8s | $0.06 |
| Prompt C (experimental) | 8.0/10 | 2.1s | $0.04 |

**Decision:** Prompt B gives the best quality → deploy it.

### 2.4 Dataset Creation & Annotation

**Datasets** are essential for evaluation and fine‑tuning. LangSmith provides tools to:
- **Import** existing datasets (CSV, JSON, or from traces).
- **Create** new datasets from scratch.
- **Annotate** (label) data manually.
- **Version** datasets for reproducibility.

**Creating a dataset from a trace:**
- Find a trace in LangSmith.
- Click **“Add to Dataset”** – it becomes a row in your dataset.
- Add annotations (e.g., correct/incorrect, score).

```python
# Conceptual – adding a trace to a dataset
# In LangSmith UI: find trace → click "Add to Dataset" → select dataset
```

### 2.5 User Feedback Integration

**User feedback** = capturing users’ reactions to your LLM responses.

**Implementation:**
- Add **thumbs up/down** buttons to your UI.
- When a user clicks, send the feedback to LangSmith via the API.

```python
# Conceptual – capturing user feedback
from langsmith import Client

client = Client()

# After receiving a response, get the trace ID
trace_id = "some_trace_id"

# User clicks thumbs up
client.create_feedback(
    trace_id=trace_id,
    key="user_satisfaction",
    score=1.0   # 1.0 = positive
)

# User clicks thumbs down
client.create_feedback(
    trace_id=trace_id,
    key="user_satisfaction",
    score=0.0   # 0.0 = negative
)
```

**What you can do with feedback:**
- See average satisfaction per prompt version.
- Filter traces by feedback score to analyse good/bad responses.
- Use feedback to improve your prompts or models.

### 2.6 Collaboration

LangSmith is designed for **team collaboration**:

| Feature | Description |
|---------|-------------|
| **Share trace links** | Copy a link to any trace and share with a team member. They see the exact same trace on their machine. |
| **Shared projects** | All team members can view and analyse traces from the same project. |
| **Prompt versioning** | Track changes to prompts over time; see who made what change. |
| **Custom dashboards** | Create and share custom dashboards with team‑specific metrics. |

**Why collaboration matters:**
- Before LangSmith, teams shared **screenshots** and **emails** to debug issues.
- With LangSmith, you share **live links** – everyone sees the exact same data, making debugging much faster.

---

## 3. Summary: LangSmith Features at a Glance

| Feature Category | What It Does | When to Use |
|------------------|--------------|-------------|
| **Observability** | Trace every component of a single run | Debugging, understanding how the system works |
| **Monitoring** | Track metrics across many runs | Production – catching performance/cost issues early |
| **Alerting** | Notify when metrics go outside limits | Production – proactive issue detection |
| **Evaluation** | Measure output quality against gold standards | Testing new versions before deployment |
| **Prompt Experimentation** | A/B test prompts to find the best one | Iterating on prompts, improving quality |
| **Dataset Creation** | Build and version test datasets | Evaluation and fine‑tuning |
| **User Feedback** | Capture user reactions to responses | Improving system based on real‑user data |
| **Collaboration** | Share traces, dashboards, and prompts | Team debugging, knowledge sharing |

---

## 4. Key Takeaways

- **LangGraph + LangSmith integration is seamless** – the graph becomes a trace, each node becomes a run.
- **Parallel execution is visualised correctly** – you can see multiple nodes running simultaneously.
- **Use `@traceable`** to trace individual functions inside nodes for even more granularity.
- **Monitoring** (aggregating across traces) is different from **observability** (analysing a single trace) – both are essential.
- **Evaluation and prompt experimentation** help you systematically improve your LLM application, not just guess.
- **User feedback** closes the loop – real users tell you what’s working and what’s not.
- **Collaboration features** make LangSmith a team‑friendly tool, not just for individual developers.


**Final advice:** Always integrate LangSmith when building complex LangGraph workflows – it helps with both debugging and learning how your system actually works.

### Useful Links
- [LangSmith](https://smith.langchain.com)

---

## 017. Observability in LangGraph | LangSmith Integration with LangGraph (21:39)

## Observability in LangGraph's based Chatbot

### What is Observability?
**Simple Definition**: Tracking and recording everything that happens in your chatbot

**What We Track**:
- User questions
- AI responses
- Token usage
- System latency
- Internal working of each component

### Why Observability Matters:
- Debug complex features (tools, RAG)
- Monitor performance
- Understand token costs
- Identify bottlenecks

---

## 🔧 Setting Up LangSmith

### Step 1: Create Account
1. Go to `smith.langchain.com`
2. Create an account
3. Login to dashboard

### Step 2: Generate API Key
```
Settings → API Keys → Create API Key
```

### Step 3: Add Environment Variables
Create a `.env` file with:

```python
LANGCHAIN_TRACING = "true"
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = "your-api-key-here"
LANGCHAIN_PROJECT = "chatbot-project"
```

---

## 💻 Code Implementation

### Basic Integration (Zero Code Changes!)

**Important**: Just adding environment variables automatically enables tracing!

```python
# Your existing code works without changes!
# LangSmith automatically captures:
# - All LLM calls
# - Token usage
# - Latency
# - Inputs/Outputs
```

### Example: Simple Chatbot Code
```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# Your regular chatbot code
llm = ChatOpenAI(model="gpt-3.5-turbo")
graph = StateGraph(...)  # Your graph structure
app = graph.compile()

# No LangSmith code needed! 
# Just set environment variables
```

---

## 📊 Understanding LangSmith Dashboard

### Project Structure:
```
Project (chatbot-project)
    └── Traces (each user turn)
         ├── Input (user message)
         ├── Output (AI response)
         ├── Token count
         ├── Latency
         └── Metadata
```

### What You See Per Trace:
- **Input**: User's exact question
- **Output**: AI's response
- **Tokens**: Input tokens + Output tokens
- **Latency**: Time taken to generate response
- **Status**: Success/Failure
- **Timing**: First token time, total execution time

---

## 🧵 Advanced: Organizing Traces with Threads

### The Problem:
Without threads, ALL conversations get mixed together

### The Solution: Add Thread ID

### Code Implementation:

**Before (No Thread Organization):**
```python
config = {
    "configurable": {"thread_id": session_id}
}
```

**After (With Thread Organization):**
```python
config = {
    "configurable": {"thread_id": session_id},
    "metadata": {
        "thread_id": session_id  # Explicit thread tracking
    },
    "run_name": "Chat Turn"  # Better readability
}
```

### Full Example:
```python
# When invoking your chatbot
session_id = "user-session-123"  # Unique per conversation

config = {
    "configurable": {
        "thread_id": session_id  # LangGraph uses this
    },
    "metadata": {
        "thread_id": session_id  # LangSmith uses this
    },
    "run_name": "Chat Turn"  # Display name in dashboard
}

response = app.invoke(input_data, config=config)
```

---

## 📊 Thread Organization Benefits

### What Changes:
1. **Individual Threads**: Each conversation has its own container
2. **Organized Traces**: All messages from one conversation stay together
3. **Easy Navigation**: Click thread → See entire conversation history
4. **Better Debugging**: Find specific conversations quickly

### Example Visualization:

**Thread 1: User Nitesh**
```
Turn 1: "Hi" → "Hello!"
Turn 2: "My name is Nitesh" → "Nice to meet you!"
Turn 3: "Who created you?" → "I was created by OpenAI"
```

**Thread 2: User Rahul**
```
Turn 1: "Hi, my name is Rahul" → "Hello Rahul!"
Turn 2: "Roadmap to study AI" → [Detailed roadmap]
```

---

## 🔍 Monitoring Features

### Token Usage Tracking
```python
# Automatically captured by LangSmith
Input Tokens: 150
Output Tokens: 250
Total Tokens: 400
Cost: $0.0008 (approx)
```

### Latency Monitoring
```python
# What you can see
Start Time: 12:34:56.123
End Time: 12:34:58.456
Duration: 2.333 seconds
First Token: 0.456 seconds
```

### Performance Metrics
- Success/failure rates
- Average response time
- Token usage patterns
- Error tracking

---

## 💡 Key Takeaways

### Why Observability Matters:
1. **Debugging**: Find exactly what went wrong
2. **Optimization**: Track token usage and latency
3. **Monitoring**: Watch system performance
4. **Analysis**: Understand user interactions
5. **Production Ready**: Essential for deploying apps

### Benefits of Thread Organization:
- Clean separation of conversations
- Better debugging experience
- Historical analysis per user
- Easier to find specific interactions

### LangSmith Features Not Covered (But Available):
- Monitoring dashboards
- Dataset creation
- Experiment tracking
- Prompt playground
- A/B testing

---

## 🚀 Future Use Cases

When adding complex features, observability helps:

1. **Tools Integration**: See which tools are called
2. **RAG Implementation**: Track document retrieval
3. **Multi-agent Systems**: Monitor agent interactions
4. **MCP Integration**: Debug complex workflows

---

## 📝 Quick Reference

### Environment Setup Checklist:
- [ ] Create LangSmith account
- [ ] Generate API key
- [ ] Add to `.env` file
- [ ] Set `LANGCHAIN_TRACING=true`
- [ ] Set project name

### Code Changes Needed:
- [ ] Add `thread_id` to config
- [ ] Add metadata with `thread_id`
- [ ] Add `run_name` for readability

### Viewing Your Data:
1. Go to LangSmith dashboard
2. Click your project
3. View traces in real-time
4. Click threads for organized view

---

## 🔄 Complete Code Example

```python
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Your chatbot setup
llm = ChatOpenAI(model="gpt-3.5-turbo")
# ... build your graph

# Create config with thread tracking
def create_config(session_id):
    return {
        "configurable": {"thread_id": session_id},
        "metadata": {"thread_id": session_id},
        "run_name": "Chat Turn"
    }

# Use in your chat function
def chat(user_input, session_id):
    config = create_config(session_id)
    response = app.invoke(
        {"messages": user_input}, 
        config=config
    )
    return response

# That's it! Everything is automatically traced
```

---

## 🎯 Summary

**Observability** = Complete visibility into your chatbot's operations

**Key Benefits**:
- Track everything automatically
- Debug with full context
- Monitor performance
- Organize conversations

**Two Main Features**:
1. **Traces**: Individual user turns
2. **Threads**: Complete conversations

**Why It Matters**: Essential for building production-ready AI applications

**Remember**: Starting simple with observability now makes complex features much easier to debug later!

---

## 018. Tools in LangGraph (34:19)

This tutorial demonstrates how to **add tools (actions)** to a LangGraph chatbot, enabling it to perform real‑world tasks like calculations, internet searches, and stock price lookups. The lecture covers:

- The need for tools (making the chatbot **actionable**, not just conversational).
- The architecture of a LangGraph workflow with tools: **LLM node** + **ToolNode** + **conditional routing** (`tools_condition`) + **loop** between LLM and tools.
- How to implement custom and pre‑built tools.
- Integration into the existing chatbot project (backend + frontend).
- Handling streaming (filtering out tool messages) and showing tool usage status.

---

## 📌 Important Pointers

| # | Concept | Explanation |
|---|---------|-------------|
| 1 | **Tools in LangGraph** | Enable the chatbot to perform actions (e.g., calculations, search, API calls) beyond text generation. |
| 2 | **ToolNode** | A pre‑built LangGraph node that manages a collection of tools and executes them when called. |
| 3 | **`tools_condition`** | A built‑in conditional edge function that routes flow to ToolNode if the LLM requests a tool call, or to END if not. |
| 4 | **Binding tools to LLM** | Use `llm.bind_tools(tools)` to give the LLM awareness of available tools. |
| 5 | **LLM–Tool loop** | The graph must loop from ToolNode back to the LLM node so that the LLM can process the tool output and potentially call more tools or give a final answer. |
| 6 | **Tool definition** | Tools can be **pre‑built** (e.g., `DuckDuckGoSearchRun`) or **custom** (using `@tool` decorator). A good docstring is essential for the LLM to understand the tool’s purpose. |
| 7 | **State** | The state still holds `messages` (conversation history), which now includes `ToolMessage`s in addition to `HumanMessage` and `AIMessage`. |
| 8 | **Streaming filtering** | When streaming to the UI, only `AIMessage` content should be displayed; `ToolMessage`s should be hidden or shown as status updates. |
| 9 | **Status container** | Use `st.status()` in Streamlit to show which tool is being used (better UX). |

---

## 1. Workflow Architecture

### Without Tools (Basic Chatbot)
```
START → chat_node (LLM) → END
```

### With Tools (Tool‑Augmented)
```
START → chat_node (LLM with tools)
           ↓ (conditional via tools_condition)
      ┌────┴────┐
      ↓         ↓
    END      ToolNode (executes the tool)
                ↓
           (loop back to chat_node)
```
The loop ensures the LLM can see the tool’s output and decide whether to call another tool or generate the final answer.

---

## 2. Code Example: Standalone Tool‑Augmented Workflow

```python
from typing import TypedDict, List, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage

# ---------- 1. Define State ----------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# ---------- 2. Define Tools ----------
# Pre‑built tool: DuckDuckGo search
search_tool = DuckDuckGoSearchRun()

# Custom tool: Calculator
@tool
def calculator(first_number: float, second_number: float, operation: str) -> float:
    """Perform basic arithmetic operations: add, subtract, multiply, divide.
    Args:
        first_number: first number
        second_number: second number
        operation: one of 'add', 'subtract', 'multiply', 'divide'
    """
    if operation == "add":
        return first_number + second_number
    elif operation == "subtract":
        return first_number - second_number
    elif operation == "multiply":
        return first_number * second_number
    elif operation == "divide":
        return first_number / second_number
    else:
        raise ValueError("Invalid operation")

# Custom tool: Stock price (using Alpha Vantage)
import requests
import os

@tool
def get_stock_price(symbol: str) -> dict:
    """Get the current stock price for a given company symbol (e.g., AAPL, TSLA).
    Returns a dict with price and metadata.
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    return response.json()

tools = [search_tool, calculator, get_stock_price]

# ---------- 3. LLM with Tools ----------
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# ---------- 4. Node Functions ----------
def chat_node(state: AgentState) -> dict:
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ---------- 5. Build Graph ----------
graph = StateGraph(AgentState)
graph.add_node("chat_node", chat_node)
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

# Edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tools_condition,  # returns "tools" or "__end__"
    {
        "tools": "tools",
        "__end__": END
    }
)
graph.add_edge("tools", "chat_node")  # loop back to LLM

workflow = graph.compile()

# ---------- 6. Run ----------
initial_state = {"messages": [HumanMessage(content="What is 2 * 3?")]}
result = workflow.invoke(initial_state)
print(result["messages"][-1].content)  # "The result of 2 * 3 is 6"
```

---

## 3. Tool Definition Best Practices

- **Pre‑built tools**: Import from `langchain_community.tools` (e.g., `DuckDuckGoSearchRun`, `WikipediaQueryRun`).
- **Custom tools**: Use the `@tool` decorator.
- **Docstring**: Essential – the LLM reads this to understand the tool’s purpose and arguments.
- **Type hints**: Help the LLM understand expected input types.

---

## 4. Integration with Existing Chatbot Project

The existing chatbot had a backend (`langraph_backend.py`) and a frontend (`streamlit_frontend.py`). To add tools:

### Backend Changes (`langraph_tool_backend.py`)

- Replace the old graph with the new one that includes `ToolNode` and the loop.
- Keep the checkpointer (`MemorySaver` or database) unchanged.
- Export the compiled `workflow` as `chatbot`.

### Frontend Changes (`streamlit_frontend.py`)

- Import the new backend (change the import statement).
- When streaming, filter messages so only `AIMessage` chunks are displayed (hide `ToolMessage` chunks).
- Optional: Use `st.status()` to show tool execution status.

**Streaming filter code snippet:**

```python
from langchain_core.messages import AIMessage, ToolMessage

# Inside the token generator:
for msg_chunk, metadata in stream:
    # Only stream AIMessage content; skip ToolMessage
    if isinstance(msg_chunk, AIMessage) and msg_chunk.content:
        yield msg_chunk.content
```

**Status container (advanced):**

```python
with st.status("Processing...", expanded=False) as status:
    # ... call the stream
    # You can update status based on metadata (e.g., tool name)
```

---

## 5. Homework Solution: Add a Custom Tool

**Task:** Add a **weather tool** that fetches the current temperature of a city using a free weather API (e.g., OpenWeatherMap).

### Step 1: Define the Tool

```python
import requests
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get the current temperature in Celsius for a given city.
    Args:
        city: Name of the city (e.g., 'London', 'Mumbai')
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        return f"The current temperature in {city} is {temp}°C."
    else:
        return f"Could not fetch weather for {city}."
```

### Step 2: Add to the Tools List

```python
tools = [search_tool, calculator, get_stock_price, get_weather]
```

### Step 3: Update the Backend and Test

```python
# Test query
initial = {"messages": [HumanMessage(content="What is the weather in Tokyo?")]}
result = workflow.invoke(initial)
print(result["messages"][-1].content)
# Output: "The current temperature in Tokyo is 22.5°C."
```

### Complete Backend Code (with weather tool)

```python
# langraph_tool_backend.py (full)
import os
import requests
from typing import TypedDict, List, Annotated
import operator
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# ---------- State ----------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# ---------- Tools ----------
search_tool = DuckDuckGoSearchRun()

@tool
def calculator(first_number: float, second_number: float, operation: str) -> float:
    """Perform basic arithmetic operations: add, subtract, multiply, divide."""
    if operation == "add":
        return first_number + second_number
    elif operation == "subtract":
        return first_number - second_number
    elif operation == "multiply":
        return first_number * second_number
    elif operation == "divide":
        return first_number / second_number
    else:
        raise ValueError("Invalid operation")

@tool
def get_stock_price(symbol: str) -> dict:
    """Get current stock price for a given company symbol (e.g., AAPL)."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    return requests.get(url).json()

@tool
def get_weather(city: str) -> str:
    """Get current temperature in Celsius for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        return f"The current temperature in {city} is {temp}°C."
    else:
        return f"Could not fetch weather for {city}."

tools = [search_tool, calculator, get_stock_price, get_weather]

# ---------- LLM ----------
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# ---------- Nodes ----------
def chat_node(state: AgentState) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ---------- Graph ----------
graph = StateGraph(AgentState)
graph.add_node("chat_node", chat_node)
tool_node = ToolNode(tools)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)
graph.add_edge("tools", "chat_node")

checkpointer = MemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)

# ---------- Optional: test ----------
if __name__ == "__main__":
    from langchain_core.messages import HumanMessage
    config = {"configurable": {"thread_id": "test"}}
    result = chatbot.invoke(
        {"messages": [HumanMessage(content="What is the weather in London?")]},
        config=config
    )
    print(result["messages"][-1].content)
```

### Frontend Changes (Filter Tool Messages)

In `streamlit_frontend.py`, replace the streaming generator with:

```python
from langchain_core.messages import AIMessage, ToolMessage

def token_generator():
    stream = chatbot.stream(input_state, config=CONFIG, stream_mode="messages")
    for msg_chunk, metadata in stream:
        if isinstance(msg_chunk, AIMessage) and msg_chunk.content:
            yield msg_chunk.content
```

---

## 6. Key Takeaways

- **Tools turn your chatbot from a passive talker into an active doer.**
- **LangGraph’s `ToolNode` and `tools_condition` simplify the wiring.**
- **The LLM–Tool loop is essential for multi‑step reasoning.**
- **Always provide clear tool docstrings** – the LLM uses them to decide when and how to call tools.
- **Filter streaming output** to show only AI messages, not raw tool outputs.
- **Use `st.status()`** for a polished UX showing tool usage.

---

## 019. How to build MCP Client using LangGraph (44:30)

summaries this agentic ai tutorial transcript in simple words with all detail, make note of all important pointers and also explain each important concepts with basic code examples

