# LLM-based Multi-agent Systems Learning Paradigms Review

[![Paper](https://img.shields.io/badge/Paper-arXiv-1a73e8.svg)](https://arxiv.org/abs/XXXX.XXXX)
![Stars](https://img.shields.io/github/stars/USC-Melady/LLM-based-Multi-agent-Systems-Learning-Paradigms-Review?style=social)


A professionally curated list of resources on **LLM-based Multi-agent Systems (LLM-MAS)**, focusing on **Multi-Agent
Reinforcement Learning (MARL)**.

We chart the shift from prompt-assembled LLM teams to **learning-driven** multi-agent systems—especially **LLM + MARL**—and propose a **Hierarchy of Emergent Intelligence**, organizing system intelligence into three levels - Individual, Collaborative, and Evolutionary.


## Survey paper

[**Survey on Learning Paradigms for LLM-based Multi-Agent Systems: Toward Evolutionary Intelligence**](https://arxiv.org/abs/XXXX.XXXX)

TODO: Authors

#### If you find this repository helpful for your work, please kindly cite our survey paper.

```bibtex
TODO
```

## ✨ Highlights

- **Hierarchy of Emergent Intelligence.** From single-agent perception/reasoning and tool use, to collaborative roles/communication/coordination, to **evolutionary** structures with meta-reasoning and continuous co-evolution.  
- **Three Paradigms.** 
  - **Prompting** for static orchestration, 
  - **Supervised Fine-Tuning** for protocol injection and specialization, 
  - **MARL** for adaptive, reward-driven collaboration.  
- **Coupling Spectrum.** How deep LLMs and RL are intertwined — from LLMs as semantic compilers for tasks/rewards to deeply co-evolving architectures.  
- **Foundations & Benchmarks.** Unify MARL foundations (credit assignment, communication, safety/constraints, scalability) with LLM-MAS, and map benchmarks spanning math/code/multimodal/collaborative/embodied tasks.  

<br>

<p>
  <img src="./assets/overview.png">
  <em> 
    Illustration of the “Hierarchy of Emergent Intelligence” in LLM systems. As capability and adaptivity increase, single LLMs evolve into agentic systems that organize roles and tools, then into multi-agent systems that coordinate through collaboration, communication, and adaptive architectures, and finally into evolutionary MAS that support structural evolution, meta-reasoning, and continuous co-evolution.
  </em>
</p>

## Taxonomy of LLM-MAS Systems
Our taxonomy includes three ascending levels of agentic complexity, each capturing a qualitatively distinct layer of reasoning, interaction, and adaptation in LLM-based systems

<p>
  <img src="./assets/taxonomy.png">
</p>


## Learning Paradigms
We examine three principal paradigms, each offering a distinct approach to engineering multiagent collaboration.

### Paradigm I: Prompting LLM-MAS
This foundational approach relies on sophisticated prompt engineering to assign agent roles, define tasks, and orchestrate communication without any updates to the underlying model weights. It serves as the baseline for constructing static, yet often highly effective, multi-agent systems.

#### Representative Works:
* MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework [\[paper\]](https://arxiv.org/abs/2308.00352)
[\[official code\]](https://github.com/FoundationAgents/MetaGPT)

* Chatdev: Communicative agents for software development. 
[\[paper\]](https://arxiv.org/abs/2307.07924)
[\[official code\]](https://github.com/OpenBMB/ChatDev)

* AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors 
[\[paper\]](https://arxiv.org/abs/2308.10848)
[\[official code\]](https://github.com/OpenBMB/AgentVerse)

* LLMArena: Assessing Capabilities of Large Language Models in Dynamic Multi-Agent Environments
[\[paper\]](https://arxiv.org/abs/2402.16499)
[\[official code\]](https://github.com/THU-BPM/LLMArena)

* ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate
[\[paper\]](https://arxiv.org/abs/2308.07201)
[\[official code\]](https://github.com/thunlp/ChatEval)

* LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion
[\[paper\]](https://arxiv.org/abs/2306.02561)
[\[official code\]](https://github.com/yuchenlin/LLM-Blender)

* A Dynamic LLM-Powered Agent Network for Task-Oriented Agent Collaboration
[\[paper\]](https://arxiv.org/abs/2310.02170)
[\[official code\]](https://github.com/SALT-NLP/DyLAN)

* CAMEL: Communicative Agents for "Mind" Exploration of Large Language Model Society
[\[paper\]](https://arxiv.org/abs/2303.17760)
[\[official code\]](https://github.com/camel-ai/camel)

* GAIA: a benchmark for General AI Assistants
[\[paper\]](https://arxiv.org/abs/2311.12983)
[\[official code\]](https://huggingface.co/gaia-benchmark)

* RoCo: Dialectic Multi-Robot Collaboration with Large Language Models[\[paper\]](https://arxiv.org/abs/2307.04738)
[\[official code\]](https://github.com/MandiZhao/robot-collab)

* ReAct: Synergizing Reasoning and Acting in Language Models
[\[paper\]](https://arxiv.org/abs/2210.03629)
[\[official code\]](https://github.com/ysymyth/ReAct)

* Reflexion: Language Agents with Verbal Reinforcement Learning
[\[paper\]](https://arxiv.org/abs/2303.11366)
[\[official code\]](https://github.com/noahshinn/reflexion)

* Tree of Thoughts: Deliberate Problem Solving with Large Language Models
[\[paper\]](https://arxiv.org/abs/2305.10601)
[\[official code\]](https://github.com/princeton-nlp/tree-of-thought-llm)

* Towards Reasoning in Large Language Models via Multi-Agent Peer Review Collaboration
[\[paper\]](https://arxiv.org/abs/2311.08152)
[\[official code\]](https://github.com/HITsz-TMG/Multi-agent-peer-review)

* Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate
[\[paper\]](https://arxiv.org/abs/2305.19118)
[\[official code\]](https://github.com/Skytliang/Multi-Agents-Debate)

* LEGO: A Multi-agent Collaborative Framework with Role-playing and Iterative Feedback for Causality Explanation Generation
[\[paper\]](https://aclanthology.org/2023.findings-emnlp.613/)

* AutoAgents: A Framework for Automatic Agent Generation
[\[paper\]](https://arxiv.org/abs/2309.17288)
[\[official code\]](https://github.com/Link-AGI/AutoAgents)

* OpenAgents: An Open Platform for Language Agents in the Wild
[\[paper\]](https://arxiv.org/abs/2310.10634)
[\[official code\]](https://github.com/xlang-ai/OpenAgents)

* SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering
[\[paper\]](https://arxiv.org/abs/2405.15793)
[\[official code\]](https://swe-agent.com/latest/)

* ProAgent: Building Proactive Cooperative Agents with Large Language Models
[\[paper\]](https://arxiv.org/abs/2308.11339)
[\[official code\]](https://github.com/PKU-Alignment/ProAgent)

* Cooperation, Competition, and Maliciousness: LLM-Stakeholders Interactive Negotiation
[\[paper\]](https://arxiv.org/abs/2309.17234)
[\[official code\]](https://github.com/S-Abdelnabi/LLM-Deliberation)

* Nested Mixture of Experts: Cooperative and Competitive Learning of Hybrid Dynamical System
[\[paper\]](https://arxiv.org/abs/2011.10605)

* Enhancing human-AI collaboration through logic-guided reasoning.[\[paper\]](https://proceedings.iclr.cc/paper_files/paper/2024/file/81b8390039b7302c909cb769f8b6cd93-Paper-Conference.pdf)

* Multi-Agent Consensus Seeking via Large Language Models
[\[paper\]](https://arxiv.org/abs/2310.20151)
[\[official code\]](https://github.com/WindyLab/ConsensusLLM-code)

* Evaluating Language Model Agency through Negotiations
[\[paper\]](https://arxiv.org/abs/2401.04536)
[\[official code\]](https://github.com/epfl-dlab/LAMEN)

* Improving Factuality and Reasoning in Language Models through Multiagent Debate
[\[paper\]](https://arxiv.org/abs/2305.14325)
[\[official code\]](https://github.com/composable-models/llm_multiagent_debate)

* Advancing Agentic Systems: Dynamic Task Decomposition, Tool Integration and Evaluation using Novel Metrics and Dataset
[\[paper\]](https://arxiv.org/abs/2410.22457)

* Self-Organized Agents: A LLM Multi-Agent Framework toward Ultra Large-Scale Code Generation and Optimization
[\[paper\]](https://arxiv.org/abs/2404.02183)
[\[official code\]](https://github.com/tsukushiAI/self-organized-agent)

* Theory of Mind for Multi-Agent Collaboration via Large Language Models
[\[paper\]](https://arxiv.org/abs/2310.10701)
[\[official code\]](https://github.com/romanlee6/multi_LLM_comm)

* API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs
[\[paper\]](https://aclanthology.org/2023.emnlp-main.187/)
[\[official code\]](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/api-bank)

* MetaAgents: Large Language Model Based Agents for Decision-Making on Teaming
[\[paper\]](https://arxiv.org/abs/2310.06500)

* Runtime verification of self-adaptive multi-agent system using probabilistic timed automata
[\[paper\]](https://journals.sagepub.com/doi/abs/10.3233/JIFS-232397)

* Welfare Diplomacy: Benchmarking Language Model Cooperation
[\[paper\]](https://arxiv.org/abs/2310.08901)
[\[official code\]](https://github.com/mukobi/welfare-diplomacy)

* Skeleton-of-Thought: Prompting LLMs for Efficient Parallel Generation
[\[paper\]](https://arxiv.org/abs/2307.15337)
[\[official code\]](https://github.com/imagination-research/sot)

* AgentCoord: Visually Exploring Coordination Strategy for LLM-based Multi-Agent Collaboration
[\[paper\]](https://arxiv.org/abs/2404.11943)
[\[official code\]](https://github.com/AgentCoord/AgentCoord)

* Social Simulacra: Creating Populated Prototypes for Social Computing Systems
[\[paper\]](https://arxiv.org/abs/2208.04024)

* MetaQA: Combining Expert Agents for Multi-Skill Question Answering
[\[paper\]](https://arxiv.org/abs/2112.01922)
[\[official code\]](https://github.com/UKPLab/MetaQA)

* AutoAct: Automatic Agent Learning from Scratch for QA via Self-Planning
[\[paper\]](https://arxiv.org/abs/2401.05268)
[\[official code\]](https://github.com/zjunlp/AutoAct)

* Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding
[\[paper\]](https://arxiv.org/abs/2401.12954)
[\[official code\]](https://github.com/suzgunmirac/meta-prompting)

* Multi-Agent Collaboration: Harnessing the Power of Intelligent LLM Agents
[\[paper\]](https://arxiv.org/abs/2306.03314)

* Avalon's Game of Thoughts: Battle Against Deception through Recursive Contemplation
[\[paper\]](https://arxiv.org/abs/2310.01320)
[\[official code\]](https://github.com/Shenzhi-Wang/recon)

* Unleashing the Emergent Cognitive Synergy in Large Language Models: A Task-Solving Agent through Multi-Persona Self-Collaboration[\[paper\]](https://arxiv.org/abs/2307.05300)
[\[official code\]](https://github.com/MikeWangWZHL/Solo-Performance-Prompting)

* MACRec: a Multi-Agent Collaboration Framework for Recommendation
[\[paper\]](https://arxiv.org/abs/2402.15235)
[\[official code\]](https://github.com/wzf2000/MACRec)

* Examining Inter-Consistency of Large Language Models Collaboration: An In-depth Analysis via Debate
[\[paper\]](https://arxiv.org/abs/2305.11595)
[\[official code\]](https://github.com/Waste-Wood/FORD)

* MAgIC: Investigation of Large Language Model Powered Multi-Agent in Cognition, Adaptability, Rationality and Collaboration
[\[paper\]](https://arxiv.org/abs/2311.08562)
[\[official code\]](https://github.com/cathyxl/MAgIC)

* Exchange-of-Thought: Enhancing Large Language Model Capabilities through Cross-Model Communication
[\[paper\]](https://arxiv.org/abs/2312.01823)
[\[official code\]](https://github.com/yinzhangyue/EoT)

* Exploring Collaboration Mechanisms for LLM Agents: A Social Psychology View
[\[paper\]](https://arxiv.org/abs/2310.02124)
[\[official code\]](https://github.com/zjunlp/MachineSoM)

* CompeteAI: Understanding the Competition Dynamics in Large Language Model-based Agents
[\[paper\]](https://arxiv.org/abs/2310.17512)
[\[official code\]](https://github.com/microsoft/competeai)

* Agents meet OKR: An Object and Key Results Driven Agent System with Hierarchical Self-Collaboration and Self-Evaluation
[\[paper\]](https://arxiv.org/abs/2311.16542)
[\[official code\]](https://okr-agent.github.io/)

### Paradigm II: Supervised Fine-Tuning for LLM-MAS
As the first tier of the learning paradigm, SFT enhances agent capabilities by fine-tuning the base LLM on curated datasets of expert behavior. This approach is instrumental for injecting domain-specific knowledge, enforcing desired interaction protocols, or imitating complex behavioral patterns

#### Representative Works:

* Enhancing Large Vision Language Models with Self-Training on Image Comprehension
[\[paper\]](https://arxiv.org/abs/2405.19716)
[\[official code\]](https://github.com/yihedeng9/STIC)

* V-STaR: Training Verifiers for Self-Taught Reasoners
[\[paper\]](https://arxiv.org/abs/2402.06457)
[\[official code\]](https://github.com/AAVSO/VStar)

* AdaSTaR: Adaptive Data Sampling for Training Self-Taught Reasoners
[\[paper\]](https://arxiv.org/abs/2505.16322)

* Self-evolving Agents with reflective and memory-augmented abilities
[\[paper\]](https://arxiv.org/abs/2409.00872)
[\[official code\]](https://github.com/codepassionor/sage)

* SELF: Self-Evolution with Language Feedback
[\[paper\]](https://arxiv.org/abs/2310.00533)

* Self-Refine: Iterative Refinement with Self-Feedback
[\[paper\]](https://arxiv.org/abs/2303.17651)
[\[official code\]](https://github.com/madaan/self-refine)

* WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning
[\[paper\]](https://arxiv.org/abs/2411.02337)
[\[official code\]](https://github.com/THUDM/WebRL)

* Reflexion: Language Agents with Verbal Reinforcement Learning
[\[paper\]](https://arxiv.org/abs/2303.11366)
[\[official code\]](https://github.com/noahshinn/reflexion)

* DYSTIL: Dynamic Strategy Induction with Large Language Models for Reinforcement Learning
[\[paper\]](https://arxiv.org/abs/2505.03209)

* Agent Workflow Memory
[\[paper\]](https://arxiv.org/abs/2409.07429)
[\[official code\]](https://github.com/zorazrw/agent-workflow-memory)

* UI-Genie: A Self-Improving Approach for Iteratively Boosting MLLM-based Mobile GUI Agents
[\[paper\]](https://arxiv.org/abs/2505.21496)
[\[official code\]](https://github.com/Euphoria16/UI-Genie)

* Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking
[\[paper\]](https://arxiv.org/abs/2403.09629)
[\[official code\]](https://github.com/ezelikman/quiet-star)

* SiriuS: Self-improving Multi-agent Systems via Bootstrapped Reasoning
[\[paper\]](https://arxiv.org/abs/2502.04780)
[\[official code\]](https://github.com/zou-group/sirius)



### Paradigm III: Multi-Agent Reinforcement Learning for LLM-MAS
Representing the most advanced learning paradigm, MARL enables agents to learn directly from environmental interaction and feedback. It is indispensable for creating truly adaptive systems that can autonomously discover and optimize complex, long-term strategies in dynamic environments. 

#### Representative Works:

* Enhancing Multi-Agent Systems via Reinforcement Learning with LLM-based Planner and Graph-based Policy
[\[paper\]](https://arxiv.org/abs/2503.10049)

* QLLM: Do We Really Need a Mixing Network for Credit Assignment in Multi-Agent Reinforcement Learning?
[\[paper\]](https://arxiv.org/abs/2504.12961)

* Verco: Learning Coordinated Verbal Communication for Multi-agent Reinforcement Learning
[\[paper\]](https://arxiv.org/abs/2404.17780)

* MARFT: Multi-Agent Reinforcement Fine-Tuning
[\[paper\]](https://arxiv.org/abs/2504.16129)
[\[official code\]](https://github.com/jwliao-ai/MARFT)

* Language-Driven Policy Distillation for Cooperative Driving in Multi-Agent Reinforcement Learning
[\[paper\]](https://arxiv.org/abs/2410.24152)

* Towards Communication Efficient Multi-Agent Cooperations: Reinforcement Learning and LLM
[\[paper\]](https://www.techrxiv.org/users/878096/articles/1259644-towards-communication-efficient-multi-agent-cooperations-reinforcement-learning-and-llm)

* ReMA: Learning to Meta-think for LLMs with Multi-Agent Reinforcement Learning
[\[paper\]](https://arxiv.org/abs/2503.09501)
[\[official code\]](https://github.com/ziyuwan/ReMA-public)

* Self-Aware Intelligent Medical Rescue Unmanned Team via Large Language Model and Multi-Agent Reinforcement Learning[\[paper\]](https://dl.acm.org/doi/10.1145/3744103.3744144)

* Safe Multi-agent Reinforcement Learning with Natural Language Constraints
[\[paper\]](https://arxiv.org/abs/2405.20018)

* Enhancing LLM QoS Through Cloud-Edge Collaboration: A Diffusion-Based Multi-Agent Reinforcement Learning Approach[\[paper\]](https://ieeexplore.ieee.org/document/10970093)

* MAS-GPT: Training LLMs to Build LLM-based Multi-Agent Systems
[\[paper\]](https://arxiv.org/abs/2503.03686)
[\[official code\]](https://github.com/MASWorks/MAS-GPT)

* FinCon: A Synthesized LLM Multi-Agent System with Conceptual Verbal Reinforcement for Enhanced Financial Decision Making
[\[paper\]](https://arxiv.org/html/2407.06567v3)
[\[official code\]](https://github.com/MXGao-A/FAgent)

* ReSo: A Reward-driven Self-organizing LLM-based Multi-Agent System for Reasoning Tasks
[\[paper\]](https://arxiv.org/abs/2503.02390)
[\[official code\]](https://github.com/hengzzzhou/ReSo)

* LAMARL: LLM-Aided Multi-Agent Reinforcement Learning for Cooperative Policy Generation
[\[paper\]](https://arxiv.org/pdf/2506.01538)

* YOLO-MARL: You Only LLM Once for Multi-Agent Reinforcement Learning
[\[paper\]](https://arxiv.org/abs/2410.03997)
[\[official code\]](https://github.com/paulzyzy/YOLO-MARL)