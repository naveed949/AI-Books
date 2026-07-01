# Chapter 1 — Introduction to Building AI Applications with Foundation Models

> **Book:** *AI Engineering: Building Applications with Foundation Models* — Chip Huyen (O'Reilly)
> **Scope of this walkthrough:** A detailed, section-by-section pass through Chapter 1. Chapter 1 is the *overview* chapter — it lightly touches concepts that later chapters explore in depth. Cross-references to later chapters are preserved inline so you know where each thread is picked up.

---

## The one-word thesis: **scale**

Huyen opens by saying that if she had one word to describe AI post-2020, it would be **scale**. Models behind ChatGPT, Gemini, and Midjourney are so large they consume a nontrivial share of the world's electricity, and we risk exhausting publicly available internet training data.

Scaling up has **two major consequences**:

1. **Models become more powerful and capable of more tasks**, enabling more applications. More people and teams can use AI to raise productivity and create value.
2. **Training large language models (LLMs) requires data, compute, and talent that only a few organizations can afford.** This gave rise to **model as a service**: a few organizations build the models and expose them for others to use.

The net effect that defines the whole book:

> **Demand for AI applications went up while the barrier to entry for building them went down.**

This turned **AI engineering — building applications on top of readily available models — into one of the fastest-growing engineering disciplines.**

Building on ML models isn't new (recommendations, fraud detection, churn prediction predate LLMs). Many principles of productionizing AI carry over, but the new generation of large-scale, readily available models brings **new possibilities and new challenges** — the focus of the book.

**Chapter roadmap:** (1) overview of foundation models, (2) successful AI use cases showing what AI is good and not-yet-good at, (3) the new AI stack — what changed, what stayed the same, and how an AI engineer differs from a traditional ML engineer.

---

## 1. The Rise of AI Engineering

Foundation models emerged from LLMs, which emerged from plain **language models**. Apps like ChatGPT and Copilot look like they appeared out of nowhere, but they are the culmination of decades of work — the first language models date to the 1950s. This section traces the path **language models → LLMs → foundation models → AI engineering**.

### 1.1 From Language Models to Large Language Models

Language models existed for a long time but only scaled up recently thanks to **self-supervision**.

#### Language models

- A **language model encodes statistical information about one or more languages** — intuitively, how likely a word is to appear in a given context. Given "My favorite color is __", an English model should predict "blue" more often than "car".
- The statistical nature of language was noticed long ago: Sherlock Holmes decoded stick figures in "The Adventure of the Dancing Men" (1905) using letter frequency (E is most common). **Claude Shannon**'s 1951 paper *"Prediction and Entropy of Printed English"* introduced concepts (like **entropy**) still used today.

**Tokens and tokenization**
- The basic unit of a language model is the **token** — a character, a word, or part of a word (like `-tion`), depending on the model.
- Example: GPT-4 breaks *"I can't wait to build AI applications"* into **nine tokens** (Figure 1-1); "can't" splits into `can` and `'t`.
- **Tokenization** = breaking text into tokens. For GPT-4, an average token ≈ ¾ of a word, so **100 tokens ≈ 75 words**.
- The set of all tokens a model can use is its **vocabulary**. Mixtral 8×7B: 32,000. GPT-4: 100,256. Tokenization method and vocab size are chosen by model developers.

**Why tokens (not words or characters)?** Three reasons:
1. Tokens let the model break words into **meaningful components** ("cooking" → "cook" + "ing").
2. Fewer unique tokens than unique words → **smaller vocabulary → more efficient** model (Chapter 2).
3. Tokens help handle **unknown words** ("chatgpting" → "chatgpt" + "ing"). Tokens balance *fewer units than words* with *more meaning than characters*.

**Two main types of language models** (differ by what context they use to predict a token):

| Type | Predicts using | Typical use today | Example |
|------|----------------|-------------------|---------|
| **Masked language model** | Context **before and after** the missing token ("fill in the blank") | Non-generative tasks: sentiment analysis, classification, code debugging (needs surrounding context) | **BERT** (Devlin et al., 2018) |
| **Autoregressive language model** | **Only preceding** tokens; predicts the next token, one after another | Text generation — the dominant, more popular choice today | GPT family |

> Convention in the book: unless stated otherwise, **"language model" means autoregressive.** (Autoregressive models are sometimes called *causal* language models.)

**Open-ended / generative.** A language model uses a fixed, finite vocabulary to build **infinite possible outputs**. A model that produces open-ended outputs is **generative** — hence **generative AI**.

**The completion machine mental model.** Given a prompt, a language model tries to complete it:
- Prompt: *"To be or not to be"* → Completion: *", that is the question."*
- Completions are **probabilistic predictions, not guaranteed correct** — this probabilistic nature is both exciting and frustrating (explored in Chapter 2).
- Completion is surprisingly powerful: **translation, summarization, coding, math, spam classification** can all be framed as completion.
  - "How are you in French is …" → "Comment ça va" (translation)
  - "Is this email likely spam? … Answer:" → "Likely spam" (classification)
- **But completion ≠ conversation.** A raw completion machine might respond to a question with another question. Making a model respond appropriately to requests is covered under **"Post-Training"** (Chapter 2 / later).

#### Self-supervision — the key to scaling

Why did language models become the center of the scaling approach (vs. object detection, recommenders, forecasting)?

> **Because language models can be trained with self-supervision, while many other models require supervision.**

- **Supervision** = training on **labeled data**, which is expensive and slow to obtain. Example: a fraud detector trained on transactions labeled "fraud"/"not fraud". The 2010s deep learning boom was supervised — **AlexNet** (Krizhevsky et al., 2012) learned to classify 1M+ ImageNet images into 1,000 categories.
- **The labeling bottleneck:** at 5¢/image, labeling 1M images ≈ **$50,000** (double it for cross-checking). Scaling to 1M categories → **$50M** in labeling alone. Some labels are far costlier (Latin translations, "does this CT scan show cancer").
- **Self-supervision:** the model **infers labels from the input data itself** — no explicit labels needed. Language modeling is self-supervised because each sequence supplies both the labels (next tokens) and the context. Example — "I love street food." yields **6 training samples** (Table 1-1):

  | Input (context) | Output (next token) |
  |---|---|
  | `<BOS>` | I |
  | `<BOS>, I` | love |
  | `<BOS>, I, love` | street |
  | `<BOS>, I, love, street` | food |
  | `<BOS>, I, love, street, food` | . |
  | `<BOS>, I, love, street, food, .` | `<EOS>` |

  `<BOS>`/`<EOS>` mark beginning/end of sequence (each usually one special token). `<EOS>` is important — it tells the model when to stop.

> **Self-supervision ≠ unsupervised learning.** Self-supervised: labels are *inferred from the input*. Unsupervised: *no labels at all.*

Because text is everywhere (books, blogs, articles, Reddit), self-supervision unlocks **massive training data**, letting language models scale into **LLMs**.

**What makes an LLM "large"?** Not a scientific term — measured by **number of parameters** (a **parameter** is a variable updated during training). More parameters → generally more capacity to learn. Historical drift:
- GPT-1 (June 2018): **117M** params — "large" at the time.
- GPT-2 (Feb 2019): **1.5B** — now 117M looked small.
- At time of writing: **100B** is considered large. Someday it'll be tiny.

**Why do larger models need more data?** More capacity to learn → needs more data to *maximize* performance. You *can* train a big model on a small dataset, but it wastes compute — a smaller model would've done as well. (The goal is maximizing performance, not matching a small model.)

### 1.2 From Large Language Models to Foundation Models

Language models are limited to **text**, but humans perceive via vision, hearing, touch, etc. To operate in the real world, AI must handle **more modalities**.

- Models are being extended to more **data modalities**: GPT-4V and Claude 3 understand images + text; some models handle video, 3D assets, protein structures. OpenAI's GPT-4V system card (2023) calls incorporating additional modalities "a key frontier in AI."
- **Foundation model** is the better term (vs. LLM) for Gemini/GPT-4V. **"Foundation"** signifies both their **importance** to AI applications and that they can be **built upon** for different needs.

**A break from AI's traditional structure.** AI research was historically split by modality:
- **NLP** → text (translation, spam detection)
- **Computer vision** → images (object detection, classification)
- **Audio** → speech recognition (STT) and synthesis (TTS)

Terminology:
- A model handling **more than one modality** = **multimodal model.**
- A **generative** multimodal model = **large multimodal model (LMM).**
- Where a language model predicts the next token from text tokens, a multimodal model predicts it from text + image (or other) tokens (Figure 1-3).

**Self-supervision works for multimodal too.** OpenAI's **CLIP** (2021) used **natural language supervision**: instead of manual labels, it harvested **400M (image, text) pairs** co-occurring on the internet — ~400× ImageNet, with no manual labeling cost. CLIP became the first model to generalize to many image-classification tasks **without additional training.**

- **CLIP is not generative** — it's an **embedding model** producing **joint embeddings** of text and images. (Embeddings = vectors that capture the meaning of the original data; detailed in "Introduction to Embedding.") Multimodal embedding models like CLIP are backbones of generative multimodal models (Flamingo, LLaVA, Gemini).

> **Book convention:** "foundation models" refers to **both large language models and large multimodal models.**

**Task-specific → general-purpose.** Previously, a sentiment model couldn't translate and vice versa. Foundation models, thanks to scale and training, are **general-purpose** — an LLM can do both sentiment analysis and translation out of the box (Figure 1-4 shows the Super-NaturalInstructions benchmark task range; Wang et al., 2022).

**Adapting general models to your task.** Out-of-the-box models may be accurate but miss brand voice, etc. Three very common **AI engineering techniques** to close the gap (each detailed later in the book):
- **Prompt engineering** — craft detailed instructions + examples.
- **Retrieval-augmented generation (RAG)** — supplement instructions with a database (e.g., customer reviews).
- **Finetuning** — further-train the model on a high-quality dataset.

**Adapting an existing model ≫ building from scratch:** roughly *"ten examples and one weekend vs. 1M examples and six months."* Foundation models make AI apps **cheaper to develop and faster to market.** How much data you need depends on the technique. Task-specific models still have benefits (often much smaller → faster and cheaper). Build-vs-buy is a **classic decision** each team must make.

### 1.3 From Foundation Models to AI Engineering

**AI engineering = building applications on top of foundation models.** People have built AI apps for a decade (ML engineering / **MLOps**). Why talk about AI engineering *now*?

> If traditional ML engineering **develops** models, AI engineering **leverages existing** ones — focusing less on modeling/training and more on **model adaptation.**

Three factors create ideal conditions for AI engineering's explosive growth:

- **Factor 1 — General-purpose AI capabilities.** Foundation models don't just do existing tasks better; they do **more tasks**. Previously impossible apps are now possible; new ones keep emerging. Because AI can write as well as (sometimes better than) humans, it can automate any task requiring communication — "pretty much everything" (emails, customer requests, contract explanations, images/videos, code, even synthesizing training data to build stronger future models). This vastly grows the user base and demand.

- **Factor 2 — Increased AI investments.** ChatGPT's success triggered a surge in VC and enterprise investment. Cheaper-to-build, faster-to-market apps → attractive ROI. (Scribd's Matt Ross: estimated AI cost dropped **two orders of magnitude** April 2022 → April 2023.) Goldman Sachs projected AI investment approaching **$100B US / $200B globally by 2025**. FactSet: **1 in 3 S&P 500** companies mentioned AI in Q2 2023 earnings calls — 3× the prior year (Figure 1-5). WallStreetZen: companies mentioning AI saw stock rise **4.6% vs. 2.4%** (causation vs. correlation unclear).

- **Factor 3 — Low entrance barrier.** The **model-as-a-service** approach (OpenAI et al.) exposes models via **APIs** — no need to host/serve infrastructure yourself; power via a single API call. AI also lowers the coding barrier: (1) AI can **write code** for non-engineers, and (2) you can work with models in **plain English**. "Anyone, and I mean anyone, can now develop AI applications." Building the *foundation models* themselves, though, remains limited to big corporations (Google, Meta, Microsoft, Baidu, Tencent), governments (Japan, UAE), and well-funded startups (OpenAI, Anthropic, Mistral). Sam Altman (Sept 2022): the biggest opportunity for most people is **adapting** these models for specific applications.

**Evidence of the surge:** Within two years, four open source AI tools (AutoGPT, Stable Diffusion Web UI, LangChain, Ollama) drew more GitHub stars than Bitcoin, on track to pass React and Vue (Figure 1-6). A LinkedIn survey (Aug 2023) found profiles adding "Generative AI," "ChatGPT," "Prompt Engineering," etc. grew **~75% per month**.

#### Why the term "AI Engineering"?

Huyen surveyed 20 practitioners; most preferred **"AI engineering."** She rejected:
- **"ML engineering"** — working with foundation models differs enough from traditional ML that "ML engineering" doesn't capture the distinction (though it's a fine umbrella for both).
- **"...Ops"** terms (MLOps, AIOps, LLMOps) — there *are* operational components, but the focus is more on **engineering/tweaking** models to do what you want.

---

## 2. Foundation Model Use Cases

The potential application space seems **endless** — "whatever use case you think of, there's probably an AI for that." Even categorizing is hard; surveys differ:
- **AWS:** customer experience, employee productivity, process optimization.
- **2024 O'Reilly survey (8 categories):** programming, data analysis, customer support, marketing copy, other copy, research, web design, art.
- **Deloitte (by value capture):** cost reduction, process efficiency, growth, accelerating innovation.
- **Gartner:** adds **business continuity** — a firm may go out of business if it doesn't adopt genAI (**7%** of 2,500 executives cited this in 2023).

**Exposure research — Eloundou et al. (2023), "GPTs are GPTs":** a task is "exposed" if AI can cut its time by ≥50%. Occupations near **100% exposure**: interpreters/translators, tax preparers, web designers, writers, mathematicians, financial quantitative analysts (Table 1-2). **No exposure**: cooks, stonemasons, athletes.

**Huyen's own study** (basis for the categories): interviewed 50 companies + read 100+ enterprise case studies; examined **205 open source AI apps** (≥500 GitHub stars). She grouped use cases into **eight categories** (Table 1-3), spanning consumer and enterprise:

| Category | Consumer examples | Enterprise examples |
|---|---|---|
| **Coding** | Coding | Coding |
| **Image & video production** | Photo/video editing, design | Presentation, ad generation |
| **Writing** | Email, social/blog posts | Copywriting/SEO; reports, memos, design docs |
| **Education** | Tutoring, essay grading | Employee onboarding, upskilling |
| **Conversational bots** | General chatbot, AI companion | Customer support, product copilots |
| **Information aggregation** | Summarization, talk-to-your-docs | Summarization, market research |
| **Data organization** | Image search, memex | Knowledge management, document processing |
| **Workflow automation** | Travel/event planning | Data extraction/entry/annotation, lead generation |

Because foundation models are general, one app can span **multiple categories** (a bot that's both companion and information aggregator; an app that extracts data from a PDF *and* answers questions about it).

**Distribution of the 205 open source apps** (Figure 1-7): small shares for education, data organization, and writing **don't mean unpopular** — those apps tend to be **enterprise/closed-source**.

**Enterprise prefers lower risk:** a16z Growth (2024) shows companies deploy **internal-facing** apps (internal knowledge management) faster than **external-facing** ones (customer-support chatbots) (Figure 1-8). Internal apps build AI expertise while limiting privacy/compliance/catastrophic-failure risk. Likewise, though foundation models are open-ended, many apps built on them stay **close-ended** (e.g., classification) because those are **easier to evaluate → easier to estimate risk.**

> The dominant future use case may surprise us — as social media surprised the early internet.

The eight use cases in detail:

### Coding
Hands-down the most popular use case (AI is good at it, and early AI engineers are coders). **GitHub Copilot** crossed **$100M ARR** two years after launch. Funding: Magic ($320M), Anysphere ($60M), both Aug 2024. Open source tools (gpt-engineer, screenshot-to-code) hit 50k stars within a year. Specialized tasks: extract structured data from pages/PDFs (AgentGPT), English→code (DB-GPT, SQL Chat, PandasAI), design/screenshot→website (screenshot-to-code, draw-a-ui), language/framework translation (GPT-Migrate), docs (Autodoc), tests (PentestGPT), commit messages (AI Commits).

**Will AI replace software engineers?** Spectrum of views — Jensen Huang (NVIDIA) predicts replacement; AWS's Matt Garman says most developers will stop *coding* (jobs change, not disappear); many engineers insist they won't be replaced. **McKinsey:** AI makes developers ~2× more productive on documentation, **25–50%** on generation/refactoring, but minimal gains on **highly complex** tasks (Figure 1-9). AI is reportedly better at **frontend** than backend. Regardless of replacement, AI makes engineers **more productive** → more done with fewer engineers; likely disrupts **outsourcing** (simpler, non-core tasks).

### Image and Video Production
AI's **probabilistic nature** makes it great for **creative** tasks. Successful startups: Midjourney (image), Adobe Firefly (photo editing), Runway/Pika Labs/Sora (video). Midjourney hit **$200M ARR** at ~1.5 years old. Half of the top-10 free Graphics & Design apps on the App Store (Dec 2023) had "AI" in their names. Common uses: **profile pictures** (Facebook banned AI profile photos in 2019; by 2023 apps offer the feature). Enterprise: ads/marketing — generate promotional images/videos, brainstorm, A/B test ads, produce seasonal/location variations (add fall colors or snow).

### Writing
AI has long aided writing (autocorrect, autocompletion). Writing is ideal for AI: **done a lot, tedious, high tolerance for mistakes** (ignore bad suggestions). LLMs are naturally good at it (trained for completion). MIT study (Noy & Zhang, 2023): among 453 professionals, ChatGPT users saw **time −40%, quality +18%**, and it **narrows the quality gap** (helps weaker writers most). Consumer uses: soften angry emails, expand bullet points, essays, fiction (interactive/branching AI books; a kids' app builds stories around words a child struggles with). Grammarly finetunes a model for fluency/clarity. **Abuse:** Amazon flooded with shoddy AI travel guides (NYT, 2023). Enterprise: sales, marketing, performance reports, cold outreach, ad copy, product descriptions (HubSpot, Salesforce). AI is **especially good at SEO** (trained on SEO-heavy internet text) — enabling **content farms** (NewsGuard, June 2023: ~400 ads from 141 brands on junk AI sites; one site produced 1,200 articles/day). A bleak possible future for internet content.

### Education
When ChatGPT goes down, students flood OpenAI's Discord. NYC Public Schools and LA Unified initially **banned** ChatGPT (cheating fears), then reversed. Instead of banning, schools can use AI to **personalize learning**: summarize textbooks, generate per-student lecture plans, adapt format (read aloud for auditory learners, animal-themed visuals, math→code). Great for **language learning** (roleplay scenarios). Duolingo (Pajak & Bicknell, 2022): of four course-creation stages, **personalization** benefits most from AI (Figure 1-10). AI can generate/evaluate quizzes and act as a **debate partner** (better at presenting multiple views than the average human). Khan Academy offers AI teaching/course assistants. Disruption: **Chegg**'s share price fell from **$28 (Nov 2022) to $2 (Sept 2024)** as students turned to AI. Framing: the risk (AI replaces skills) is mirrored by the opportunity (AI as a **tutor** to learn any skill).

### Conversational Bots
Versatile: find info, explain concepts, brainstorm; companion/therapist; emulate personalities. **Digital girlfriends/boyfriends** became popular fast; many spend more time with bots than humans. Research: groups of bots can **simulate a society** (Park et al., 2023). Enterprise: **customer support** bots (cut costs, faster responses), **product copilots** (guide through taxes, insurance claims, policies). Beyond text: **voice** (Google Assistant, Siri, Alexa) and **3D** bots (games, retail, marketing) — e.g., **smart NPCs** (NVIDIA's Inworld, Convai) that make games like The Sims/Skyrim more dynamic.

### Information Aggregation
We're overwhelmed by emails, Slack, news. Salesforce (2023): **74%** of genAI users use it to distill/summarize. Consumer: **talk-to-your-docs** (contracts, disclosures, papers), summarize websites/research. Enterprise: aggregation/distillation makes orgs **leaner** (less middle-management burden). Instacart's internal prompt marketplace's most popular template: **"Fast Breakdown"** — summarize meetings/emails/Slack into facts, open questions, action items (auto-inserted into a project tracker). Aggregation pairs naturally with data organization.

### Data Organization
We keep producing more data (photos, videos, logs, PDFs — unstructured/semi-structured). AI helps **organize it for search**: auto-generate text descriptions for images/videos, match text queries to visuals (Google Photos), even **generate** images when none match (Google Image Search). AI is strong at **data analysis** (visualizations, outliers, forecasts). Enterprise: extract **structured info from unstructured data** — from credit cards, licenses, receipts, tickets, email footers, to contracts/reports/charts. The **IDP (intelligent data processing)** industry is projected to reach **$12.81B by 2030** (32.9% CAGR).

### Workflow Automation
The ultimate goal — automate as much as possible. Consumer: booking restaurants, refunds, trip planning, forms. Enterprise: lead management, invoicing, reimbursements, customer requests, data entry. Exciting case: AI **synthesizing data** to improve the models themselves (human-in-the-loop label improvement) — data synthesis in Chapter 8. Many tasks need **external tools** (search, phone, calendar). **AIs that can plan and use tools are called agents** — enormous interest, potentially making everyone far more productive. **Agents are a central topic in Chapter 6.**

> Not all applications *should* be built → next section.

---

## 3. Planning AI Applications

> "It's easy to build a cool demo with foundation models. It's hard to create a profitable product."

If you're learning/having fun, just build. If it's for a living, step back and consider **why** and **how**.

### 3.1 Use Case Evaluation

First ask **why** you're building it. Like most business decisions, it's a response to **risks and opportunities**, ordered high→low risk:

1. **Existential risk** — "If you don't, AI competitors make you obsolete." Highest priority. Common in document processing / information aggregation (financial analysis, insurance, data processing) and creative work (advertising, web design, image production). (Gartner 2023: 7% cited **business continuity**.)
2. **Missed opportunity** — "You'll miss profit/productivity gains." Most companies adopt AI for opportunity: cheaper acquisition (copy, descriptions, visuals), better retention (support, personalization), plus lead gen, internal comms, market research, competitor tracking.
3. **Fear of being left behind** — Unsure where AI fits, but don't want to lag. Don't chase every hype train, but waiting too long has killed companies (**Kodak, Blockbuster, BlackBerry**). Investing to understand a transformational tech is reasonable if you can afford it (often R&D at bigger firms).

Then ask whether **you must build it yourself.** If AI is existential → keep it **in-house** (don't outsource to a competitor). If it's for profit/productivity → many **buy** options may save time/money and perform better.

#### The role of AI and humans in the application

Apple's guidance offers three axes:

- **Critical vs. complementary** — Does the app work without AI? Face ID *needs* AI (critical); Gmail works without Smart Compose (complementary). **The more critical AI is, the more accurate/reliable it must be** (people forgive mistakes when AI isn't core).
- **Reactive vs. proactive** — Reactive responds to user requests (chatbot); proactive fires on opportunity (Google Maps traffic alerts). Reactive usually needs to be **fast**; proactive can be **precomputed** (latency matters less) but has a **higher quality bar** (unwanted low-quality proactivity feels intrusive).
- **Dynamic vs. static** — Dynamic updates continually with feedback (Face ID adapts as faces change; per-user finetuning; ChatGPT memory); static updates periodically (object detection in Google Photos updates on app upgrades; often one shared model for a group).

**Role of humans** (customer-support chatbot example):
- AI drafts responses that human agents reference,
- AI handles simple requests and routes complex ones to humans,
- AI answers everything directly.

Involving humans is **human-in-the-loop**. Microsoft's (2023) **Crawl–Walk–Run** framework for ramping automation:
1. **Crawl** — human involvement mandatory.
2. **Walk** — AI interacts directly with **internal** employees.
3. **Run** — increased automation, potentially **direct interaction with external users**.

The human role shifts as quality improves (e.g., once agents accept 95% of AI suggestions verbatim for simple requests, let customers hit AI directly for those).

#### AI product defensibility

If you sell AI apps as standalone products, consider **moats**. The low entry barrier cuts both ways — easy for you means easy for competitors.

- **Risk of being subsumed by the model:** you're providing a *layer* on top of foundation models. If the base model expands (e.g., ChatGPT gets great at PDF parsing at scale), your layer may become obsolete. (A PDF-parser might still make sense on **open source** models for in-house-hosting users.)
- A VC partner: many startups' entire products **could be a feature** of Google Docs / Microsoft Office — what stops Google from cloning it in two weeks with three engineers?

**Three competitive advantages in AI: technology, data, distribution.**
- **Technology** — with foundation models, most companies' core tech is **similar**.
- **Distribution** — advantage likely belongs to **big companies**.
- **Data** — nuanced: big companies have more existing data, **but** a startup that reaches market first and gathers **usage data** can make data its moat. Even when usage data can't train models directly, it yields invaluable insight into user behavior and product gaps (guiding data collection/training).

Encouragement: many winners started as a feature bigger players overlooked — **Calendly** (vs. Google Calendar), **Mailchimp** (vs. Gmail), **Photoroom** (vs. Google Photos).

### 3.2 Setting Expectations

Define **what success looks like**. The most important metric is **business impact.** For a support chatbot:
- % of customer messages to automate,
- how many more messages you can process,
- how much faster you can respond,
- how much human labor you save.

But answering more messages ≠ happy users — track **customer satisfaction and feedback** ("User Feedback" covers feedback-system design, Chapter 10).

Set a **usefulness threshold** — how good it must be to be useful — before shipping. Metric groups:
- **Quality metrics** — response quality.
- **Latency metrics** — **TTFT** (time to first token), **TPOT** (time per output token), total latency. Acceptable latency is use-case dependent (if humans currently take an hour median, anything faster may suffice).
- **Cost metrics** — cost per inference request.
- **Others** — interpretability, fairness.

(If you're unsure which metrics — the rest of the book covers many.)

### 3.3 Milestone Planning

Plan how to reach the goals; the path depends on where you start. **Evaluate off-the-shelf models first** — stronger models mean less work (if the target is automating 60% of tickets and an off-the-shelf model already does 30%, you need less effort than starting from zero). **Goals may change after evaluation** — you might learn the effort to hit the threshold exceeds the return, and drop it.

**The last-mile challenge.** Initial success is misleading: base capabilities make a fun demo quick, but a **good demo doesn't promise a good product.** A weekend for a demo; months or years for a product.
- UltraChat (Ding et al., 2023): *"the journey from 0 to 60 is easy, whereas progressing from 60 to 100 becomes exceedingly challenging."*
- LinkedIn (2024): **1 month to reach 80%** of the desired experience, but **4 more months to surpass 95%** — much of it fighting product kinks and **hallucinations**, with each additional 1% painfully slow.

### 3.4 Maintenance

Planning doesn't stop at launch. AI's **fast pace of change** adds a maintenance challenge — building on foundation models means "riding this bullet train."

- **Good changes still cause friction:** longer context lengths, better outputs, cheaper/faster **inference** (Figure 1-11: cost vs. MMLU performance, 2022–2024; MMLU = Massive Multitask Language Understanding, Hendrycks et al., 2020). You must continually run **cost-benefit analysis** — today's best option can become tomorrow's worst (build in-house to save money, then providers halve prices; tailor around a third-party that then goes bankrupt).
- **Easier changes:** as providers converge on the same API, **swapping models** gets easier — but each model has quirks; without **versioning and evaluation** infrastructure, migration is painful.
- **Harder changes — regulation:** AI is a national-security issue in many countries; compute/talent/data are heavily regulated. GDPR compliance was estimated at **$9B** for businesses; compute availability can change overnight (US Oct 2023 Executive Order); a banned GPU vendor is real trouble.
- **Potentially fatal changes — IP:** rules around IP and AI usage are evolving. If your product is built on a model trained on others' data, can you be sure you keep your IP? IP-heavy firms (e.g., game studios) hesitate for this reason.

---

## 4. The AI Engineering Stack

The hype/FOMO is overwhelming — instead of chasing shifting sand, focus on **fundamental building blocks.** AI engineering **evolved out of ML engineering**: when a company first tries foundation models, its existing ML team usually leads. Some firms treat AI engineering = ML engineering (Figure 1-12); others write separate AI-engineering job descriptions (Figure 1-13). Either way, the roles **overlap significantly** — ML engineers can add AI engineering skills, and some AI engineers have **no prior ML experience.**

### 4.1 Three Layers of the AI Stack

Any AI application stack has **three layers**; you typically start at the **top** and move down as needed (Figure 1-14):

1. **Application development** — with models readily available, anyone can build apps. Most action in the last two years, still rapidly evolving. Involves **good prompts + necessary context**, **rigorous evaluation**, and **good interfaces**.
2. **Model development** — tooling for **modeling, training, finetuning, inference optimization**, plus **dataset engineering** (data is central), and **rigorous evaluation**.
3. **Infrastructure** — at the bottom: **model serving, managing data and compute, monitoring.**

**Ecosystem data (March 2024):** Huyen searched GitHub for AI repos with ≥500 stars — **920 repositories** (Figure 1-15). Big jump in AI tooling in 2023 after Stable Diffusion and ChatGPT; the largest 2023 increases were in **applications and application development.** **Infrastructure grew least** — expected, because core infra needs (resource management, serving, monitoring) **remain the same** even as models/apps change.

**What stays the same:** For enterprise, AI apps still must **solve business problems**, so mapping **business metrics ↔ ML metrics** still matters. You still need **systematic experimentation** (classical ML tunes hyperparameters; with foundation models you experiment with models, prompts, retrieval algorithms, **sampling variables** [Chapter 2], etc.). You still make models **faster/cheaper** and still build a **feedback loop** to improve with production data. **A decade of ML-engineering wisdom still applies** — but with new innovations layered on top.

### 4.2 AI Engineering vs. ML Engineering

Three high-level differences from traditional ML engineering:

1. **Model adaptation over training.** Without foundation models you train your own; with AI engineering you use someone else's model → **less modeling/training, more adaptation.**
2. **Bigger, heavier models.** More compute, higher latency → more pressure for **efficient training and inference optimization**, and more need for engineers who can work with **GPUs and big clusters** ("knows 10 GPUs, not 1,000 GPUs").
3. **Open-ended outputs.** More flexible (more tasks) but **harder to evaluate** → **evaluation becomes a much bigger problem.**

> In short: AI engineering is **less about model development, more about adapting and evaluating models.**

**Model adaptation — two categories** (by whether they update model weights):
- **Prompt-based techniques** (incl. prompt engineering): **do not** update weights — adapt via instructions/context. Easier to start, needs less data, lets you try many models. May be insufficient for complex tasks or strict performance requirements.
- **Finetuning:** **updates** weights — adapt the model itself. More complex, needs more data, but can significantly improve quality/latency/cost and enable things impossible otherwise (a new task not seen in training).

#### Model development (the traditionally-ML layer)

Three responsibilities (evaluation is discussed under app development):

- **Modeling and training** — devising an architecture, training, finetuning. Tools: TensorFlow, Hugging Face Transformers, PyTorch. Requires ML knowledge (algorithms: clustering, logistic regression, decision trees, collaborative filtering; architectures: feedforward, recurrent, convolutional, transformer; concepts: gradient descent, loss functions, regularization). **With foundation models, ML knowledge is a nice-to-have, not a must-have** — many successful builders don't care about gradient descent — but it's still valuable for expanding your toolset and troubleshooting.

  > **Sidebar — training vs. pre-training vs. finetuning vs. post-training.** Training always changes weights, but not all weight changes are training (**quantization** reduces precision, changing weights, but isn't training).
  > - **Pre-training** — training from scratch (randomly initialized weights); for LLMs, usually text completion. The most **resource-intensive** phase by far (up to **98%** of InstructGPT's compute/data). A small mistake is costly; an art few practice, and pre-training experts are heavily sought after.
  > - **Finetuning** — continue training a previously trained model (weights inherited); needs fewer resources than pre-training.
  > - **Post-training** — training after pre-training. Conceptually the same as finetuning; the terms often differ by **who** does it: **post-training** by *model developers* (e.g., OpenAI makes a model follow instructions before release); **finetuning** by *application developers* (you adapt an OpenAI model to your needs). Pre-training and post-training form a **spectrum** with similar processes/tooling (Chapters 2 and 7).
  > - Note: **prompt engineering is not training.** Feeding journal entries into ChatGPT's context is prompt engineering, not training (a common misuse of the terms).

- **Dataset engineering** — curating, generating, annotating data for training/adaptation. Traditional ML is mostly **close-ended** (predefined outputs, e.g., spam/not-spam) and **tabular**; foundation models are **open-ended** and **unstructured**. Annotating open-ended data is much harder (writing an essay vs. labeling spam), so annotation is a **bigger challenge**. AI-engineering data work centers on **deduplication, tokenization, context retrieval, quality control** (removing sensitive/toxic data) — Chapter 8. Since models are becoming commodities, **data is the main differentiator.** Data need scales with technique: **scratch > finetuning > prompt engineering.**

- **Inference optimization** — making models **faster and cheaper**. Always mattered; **even more important** now given foundation models' cost/latency. Challenge: **autoregressive** generation is sequential (10 ms/token → 1 s for 100 tokens), while typical internet apps expect **~100 ms** latency. Techniques (**quantization, distillation, parallelism**) covered in Chapters 7–9.

  **Table 1-4 — model development, traditional ML vs. foundation models:**

  | Category | Traditional ML | Foundation models |
  |---|---|---|
  | Modeling & training | ML knowledge required (train from scratch) | ML knowledge nice-to-have, not must-have* |
  | Dataset engineering | Feature engineering, tabular data | Less feature engineering; more dedup, tokenization, context retrieval, quality control |
  | Inference optimization | Important | Even more important |

  *Many would dispute this, insisting ML knowledge is a must-have.

#### Application development (where differentiation now lives)

With traditional ML, proprietary **model quality** differentiates. With foundation models, **many teams share the same model**, so **differentiation comes from application development.** Three responsibilities:

- **Evaluation** — mitigating risks and uncovering opportunities; needed throughout adaptation (select models, benchmark progress, decide deployment readiness, detect production issues/opportunities). **Harder** than in traditional ML because of open-endedness and expanded capabilities:
  - Close-ended tasks (fraud) have **ground truth** to compare against; chatbots have **so many valid responses** that an exhaustive ground-truth list is impossible.
  - Many adaptation techniques complicate evaluation. **Gemini vs. ChatGPT (Dec 2023):** Google claimed a higher MMLU score, but had used **CoT@32** (32 examples) for Gemini vs. **5 examples** for ChatGPT. **At equal 5 examples, GPT-4 performed better** (Table 1-5). The same prompt technique took Gemini Ultra from **83.7% → 90.04%.** Evaluation challenges: Chapter 3.

- **Prompt engineering and context construction** — get desired behavior **from input alone**, without changing weights. It's not just telling the model what to do, but providing **context and tools**, and for long/complex tasks a **memory management system** to track history. Prompt engineering: Chapter 5; context construction: Chapter 6.

- **AI interface** — how end users interact with the app. Before foundation models, only resource-rich orgs built AI, usually **embedded** in existing products (fraud detection in Stripe/Venmo/PayPal; recommenders in Netflix/TikTok/Spotify). Now anyone can ship **standalone** products (ChatGPT, Perplexity) or **embed/plug-in** (Copilot in VSCode, Grammarly browser extension, Midjourney in Discord). Popular interfaces:
  - Standalone web/desktop/mobile apps,
  - Browser extensions,
  - Chatbots in Slack/Discord/WeChat/WhatsApp,
  - Plug-ins/add-ons via product APIs (VSCode, Shopify, Microsoft 365) — also usable by **agents** (Chapter 6).

  Interfaces can also be **voice-based** or **embodied** (AR/VR). New interfaces = **new ways to collect feedback**; conversational interfaces make giving feedback easy (natural language) but the feedback **harder to extract** (Chapter 10). (Tools like Streamlit, Gradio, Plotly Dash are common for AI web apps.)

  **Table 1-6 — app development importance, traditional ML vs. foundation models:**

  | Category | Traditional ML | Foundation models |
  |---|---|---|
  | AI interface | Less important | Important |
  | Prompt engineering | Not applicable | Important |
  | Evaluation | Important | More important |

### 4.3 AI Engineering vs. Full-Stack Engineering

The rising emphasis on **application development and interfaces** brings AI engineering closer to **full-stack development.** ML engineering was traditionally **Python-centric**; now there's growing **JavaScript** support (LangChain.js, Transformers.js, OpenAI's Node library, Vercel's AI SDK) to attract frontend engineers.

More AI engineers now come from **web/full-stack** backgrounds; their edge over traditional ML engineers is turning **ideas into demos fast**, getting feedback, and iterating.

**Reversed workflow (Figure 1-16, after Shawn Wang's "The Rise of the AI Engineer," 2023):**
- **Traditional ML:** gather data → train model → build product **last**.
- **AI engineering:** build the **product first**, invest in data/models **only once the product shows promise.**

Consequently, AI engineers are **much more involved in product decisions** than ML engineers traditionally were (model and product development used to be disjointed).

> Aside: "AI engineering is just software engineering with AI models thrown in the stack." (Anton Bacaj)

---

## 5. Summary

The chapter served two purposes: (1) explain the **emergence of AI engineering** as a discipline enabled by foundation models, and (2) give an **overview of the process** to build applications on them. As the overview chapter, it only lightly touched concepts the rest of the book develops.

Key threads:
- **Evolution:** language models → **LLMs** (via **self-supervision**) → **foundation models** (adding modalities) → **AI engineering.**
- **Growth driver:** emerging capabilities unlock many **applications** (consumer + enterprise); we're still early, with much more to build.
- **Should you build it?** An often-overlooked but crucial question, alongside major considerations (risk/opportunity, human role, defensibility, expectations, milestones, maintenance).
- **Continuity + change:** AI engineering evolved from ML engineering — many principles carry over, but new challenges (adaptation, scale/efficiency, evaluation of open-ended outputs) and solutions arise.
- **The stack:** application development, model development, infrastructure — and how each shifts from ML engineering.

The community's collective energy is impossible to keep up with; the more overwhelming the space, the more valuable a **framework** to navigate it — which the book aims to provide, starting next with the **foundation models** themselves (Chapter 2).

---

## Key terms quick-reference

| Term | Meaning |
|---|---|
| **Language model** | Encodes statistical info about language; predicts likely tokens in context. |
| **Token / tokenization / vocabulary** | Basic unit (char/word/subword) / process of splitting text / the full token set a model uses. |
| **Masked vs. autoregressive LM** | Fill-in-the-blank (both-side context, e.g. BERT) vs. next-token prediction (preceding context, e.g. GPT). |
| **Generative / open-ended** | Produces infinite outputs from a finite vocabulary. |
| **Supervision / self-supervision / unsupervised** | Labeled data / labels inferred from input (enables scaling) / no labels. |
| **Parameter** | A variable in the model updated during training; rough proxy for size/capacity. |
| **Foundation model** | Large, general-purpose model (LLM or LMM) that can be built upon. |
| **Multimodal model / LMM** | Handles >1 modality / generative multimodal model. |
| **Embedding model (e.g., CLIP)** | Produces vectors capturing meaning; not generative. |
| **Model as a service** | Models exposed via APIs for others to build on. |
| **AI engineering** | Building applications on top of foundation models. |
| **Prompt engineering / RAG / finetuning** | Three adaptation techniques (no weight change / add retrieval context / update weights). |
| **Pre-training / finetuning / post-training** | Train from scratch / continue training / train after pre-training (by model developers). |
| **Human-in-the-loop / Crawl–Walk–Run** | Humans in AI decisions / Microsoft's staged automation framework. |
| **Usefulness threshold** | How good the app must be to be useful before shipping. |
| **TTFT / TPOT** | Time to first token / time per output token (latency metrics). |
| **Inference optimization** | Making models faster/cheaper (quantization, distillation, parallelism). |
| **The three AI stack layers** | Application development, model development, infrastructure. |

*Chapters referenced for deeper dives: probabilistic nature & foundation models (2), evaluation (3), prompt engineering (5), agents & context construction (6), inference optimization (7–9), dataset engineering & data synthesis (8), user feedback (10).*
