
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

## 4. LangChain Vs LangGraph (01:27:28)

summaries this agentic ai tutorial transcript in simple words with all detail, make note of all important pointers and also explain each important concepts with basic code examples