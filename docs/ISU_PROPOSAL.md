# Red Team Kitchen × Illinois State University
## Research Collaboration Proposal
### JemmaBrain — Predicted Neural Response Platform

**Proposed by:** Alexios Bluff Mara LLC (Soumit Lahiri)  
**Contact:** soumitlahiri@philanthropytraders.com  
**Date:** Spring 2026  
**ISU Colors:** Red & White — a perfect match

---

## The Pitch in One Paragraph

We've built a tool that takes any short video, audio clip, or text and predicts which regions of the brain would activate in a typical viewer — then explains those predictions at three different levels: plain language for a high school student, accessible science for the public, and full clinical detail for a neurologist. We'd like Illinois State University to be the first academic institution to use it, help validate it, and co-develop curriculum around it. We call it **JemmaBrain**, and it lives at **brain.redteamkitchen.com**.

---

## What We're Proposing

### For ISU Students and Faculty (Immediately Available)
- **Free Researcher-tier access** for all ISU students and faculty (soumitty@gmail.com or isu.edu email)
  - 20 jobs/hour (vs 1/hour for guests)
  - Expert 31B model for maximum narration depth
  - Priority queue position (processed before public submissions)
- **Discord server access** — join the Red Team Kitchen Discord to submit and see results in real time
- **3D brain viewer** — interactive cortical surface showing predicted BOLD response at `brain.redteamkitchen.com`

### For ISU Courses (Curriculum Integration)
We propose JemmaBrain as a hands-on tool for:

| Course | Department | How it fits |
|---|---|---|
| Intro to Neuroscience | Biology/Psychology | Visualize what "fMRI" means interactively |
| Cognitive Psychology | Psychology | Test predictions about attention/memory stimuli |
| Research Methods | Psychology | Discuss model-based prediction vs. ground-truth fMRI |
| Media & Communication | School of Communication | Analyze how different content styles engage the brain |
| Machine Learning | Computer Science | Study TRIBE v2 architecture and Gemma 4 narration |
| Science Communication | Any | Use the three-tier narration as a template for writing to different audiences |

### For ISU Research (Ongoing Collaboration)
1. **Validation study:** Compare TRIBE v2 predictions against ISU's actual behavioral data for 20–50 video clips. Does the model correctly rank which clips are more cognitively engaging?
2. **Audience comprehension study:** Do the three narration tiers actually communicate at the right level? 50 participants across student / public / expert groups rate understanding.
3. **Content analysis:** What brain network patterns characterize different educational video styles (lecture, demonstration, narrative)?
4. **Co-authorship:** Any publishable finding becomes a joint paper. Target journals: *NeuroImage*, *Frontiers in Neuroscience*, *Behavior Research Methods*

---

## What JemmaBrain Does (Technical Overview for Faculty)

### The Pipeline
```
Video/audio clip (any media)
        ↓
TRIBE v2 (Gallant Lab, UC Berkeley)
  → Predicts BOLD z-scores across 20,484 cortical vertices
  → Output: time-series of brain activation (2 Hz, fsaverage5 surface)
        ↓
Multi-atlas analysis (Harvard-Oxford, Jülich, Yeo-7 networks)
  → Identifies which brain networks activated, when, how much
        ↓
Gemma 4 (Google's open model, running locally)
  → Three narrations: Student · Public · Expert
        ↓
Three.js 3D viewer
  → Interactive cortical surface with animated BOLD heatmap
  → Yeo-7 network overlay, temporal playback, ROI detail
```

### Key Facts for IRB/Methods Sections
- **TRIBE v2** is a group-average model trained on 25 subjects watching natural movies (Cichy et al. / Gallant Lab). It predicts *expected* brain response for a typical viewer — not individual fMRI data.
- **No human subjects are needed** for the core tool (it's a model). Any study using ISU participants would require standard ISU IRB approval.
- **CC-BY-NC 4.0 license** — TRIBE v2 is for non-commercial use. Academic research qualifies.
- **Privacy:** No submitted video is stored beyond 7 days. No PII is collected. Results are pseudonymous job IDs.

---

## Why ISU is the Right Partner

### Academic Alignment
- ISU's Department of Psychology has active research programs in cognitive neuroscience, learning & memory, and educational psychology
- ISU's School of Communication studies media effects — our tool predicts neural engagement with any media content
- ISU's School of Information Technology can contribute to the engineering side (distributed systems, cloud computing)

### Geographic & Strategic Alignment
- ISU is in Normal, IL — 130 miles from Chicago — same Midwest ecosystem
- ISU's red color scheme is literally our brand (Red Team Kitchen)
- ISU participates in the Illinois Research Consortium — connection to Northwestern, UIC, UIUC

### Competitive Advantage for ISU
Being the **first university to validate and deploy** AI-predicted fMRI analysis creates:
- Publication opportunities (novel validation data)
- Grant leverage (NSF, NIH cite real-world adoption)
- Student recruitment story ("we do cutting-edge neural AI here")
- Media coverage opportunity (local and national science press)

---

## Proposed Partnership Tiers

### Tier 1: Observer (Month 1, Free)
- 5 ISU faculty/student accounts with Researcher-tier access
- Join our Discord server
- Informal feedback on usability and narration quality
- **Ask:** Just use it and tell us what's wrong

### Tier 2: Collaborator (Month 2–3, Free)
- Up to 50 ISU accounts (all @isu.edu addresses)
- Structured feedback via a 10-question monthly survey
- One joint "lunch & learn" session (virtual or in-person in Normal)
- **Ask:** Help us recruit 20 participants for the audience comprehension study

### Tier 3: Research Partner (Month 3–6, Revenue Share)
- Unlimited ISU accounts
- Co-authorship on validation paper
- ISU logo on our website, Discord, and hackathon submission
- ISU faculty member added as advisor to Alexios Bluff Mara LLC (equity or honorarium TBD)
- **Ask:** IRB-approved behavioral study using JemmaBrain as a research tool

---

## Budget & Resources

### What We Contribute (Alexios Bluff Mara LLC)
- RTX 5090 compute (estimated value: $0.08–$0.35/job equivalent)
- Software development time
- GCP cloud infrastructure
- Gemma 4 API access
- All ongoing maintenance

### What We Ask ISU to Consider
- Faculty time for collaboration (no budget required initially)
- IRB application submission (standard process, no cost to us)
- Participant recruitment for validation studies (ISU has participant pools)
- Optional: Small course integration stipend (~$500 for faculty developing curriculum module)

### Grant Opportunities We Can Pursue Together
| Grant | Amount | Timeline | Role |
|---|---|---|---|
| NSF SBIR Phase I (AI1/AI3) | $275K | Apply Aug 2026 | Alexios Bluff Mara LLC as lead, ISU as subcontractor |
| NIH R21 Exploratory Research | $275K | Apply Jun 2026 | ISU PI with Alexios Bluff Mara as industry partner |
| Illinois Innovation Network | $50K | Apply Apr 2026 | Joint ISU + startup application |
| Google Research Credits | $5K | Apply immediately | For GCP compute, free |

---

## Immediate Next Steps

1. **Meeting:** 30-minute video call with 1–2 interested ISU faculty (any of the above departments)
2. **Pilot accounts:** Issue 5 Researcher-tier Discord accounts to ISU participants
3. **Demo session:** I come to Normal (or Zoom) and walk through a live JemmaBrain analysis on an ISU-chosen video clip
4. **MOU:** Simple 1-page Memorandum of Understanding establishing non-exclusive collaboration terms

---

## Contact

**Soumit Lahiri**  
Alexios Bluff Mara LLC  
soumitlahiri@philanthropytraders.com  
soumitty@gmail.com  
Discord: [Red Team Kitchen server link]  
Website: redteamkitchen.com  
Tool: brain.redteamkitchen.com

*Red Team Kitchen: where bold ideas get cooked. 🔴*
