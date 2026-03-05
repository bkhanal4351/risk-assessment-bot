# EPA Risk Assessment Bot -Demo & Presentation Guide

This document highlights features added **after** the initial codebase, with example
questions designed to showcase each one during a live demo. We wrote this guide for
ourselves so the whole team can run a consistent demo.

---

## 1. Typo-Tolerant Search (Damerau-Levenshtein)

The bot handles misspellings and transposed letters using a two-layer fuzzy
matching system -prefix overlap first, then Damerau-Levenshtein edit distance
as a fallback.

| Try typing…                          | Matches to…                        | Why it works                         |
|--------------------------------------|------------------------------------|--------------------------------------|
| `total farud`                        | All fraud-related records          | "farud" → "fraud" (1 transposition)  |
| `hazadrous waste`                    | Hazardous Waste Mishandling        | "hazadrous" → "hazardous" (DL ≤ 2)  |
| `chemical spil response`             | Chemical Spill Response Delay      | "spil" → "spill" (DL = 1)           |
| `whistleblowwer retaliation`         | Whistleblower Retaliation          | "whistleblowwer" → "whistleblower"  |
| `enviromental justice`               | Environmental Justice Failure      | "enviromental" → "environmental"    |
| `procurment fraud`                   | Procurement Fraud                  | "procurment" → "procurement"        |
| `ergonmic injury`                    | Ergonomic Injury                   | "ergonmic" → "ergonomic"            |
| `phising attack`                     | Phishing Attack                    | "phising" → "phishing"              |

---

## 2. Aggregate / Statistical Questions

When the bot detects aggregate keywords (e.g., "how many", "total", "list all",
"breakdown"), it switches to a summary format -grouped counts by title instead
of the single Best Match layout.

| Example question                                  | What to look for                          |
|---------------------------------------------------|-------------------------------------------|
| `How many high risks are there?`                  | Count of high-risk records                |
| `Total fraud`                                     | All fraud records grouped with counts     |
| `List all medium risks`                           | Titles grouped by Medium risk level       |
| `Show me the breakdown by risk level`             | Counts per level (Critical/High/Med/Low)  |
| `How many risks have Detective controls?`         | Count of detective-type controls           |
| `What percentage of risks are critical?`          | Statistics across the registry            |
| `Which level has the most risks?`                 | Comparative counts                        |
| `List every high risk with its primary control`   | Full enumeration, no Best Match format    |

---

## 3. Relevance Scoring & Confidence Labels

Every response shows a colored relevance badge. We can demo the range:

| Example question                              | Expected confidence | Why                                      |
|-----------------------------------------------|---------------------|------------------------------------------|
| `What controls exist for Hazardous Waste Mishandling?` | **High** (green)    | Exact title match, strong similarity     |
| `Tell me about Chemical Spill Response Delay`          | **High** (green)    | Direct title match                       |
| `workplace safety concerns`                            | **Moderate** (orange) | Broad topic, multiple partial matches  |
| `What about climate change policy?`                    | **Low** (red)       | Not a direct registry topic              |

Thresholds: High ≥ 0.6 · Moderate ≥ 0.4 · Low < 0.4

---

## 4. Source Citations

Every answer includes an expandable "Sources from Registry" section showing:
- Rank, CSV row number, risk title, and cosine similarity score

**Demo tip:** We click the expander after any answer to show the audience where
the data came from -- this reinforces transparency and traceability.

---

## 5. Feedback Loop (Thumbs Up / Down)

Each assistant response has thumbs-up and thumbs-down buttons.

**Demo flow:**
1. Ask: `Tell me about vendor lock-in`
2. Click 👎 on the response
3. Type a comment: "Missing details about secondary controls"
4. Submit -feedback is saved to `feedback.csv`
5. Ask another question -the bot now sees past negative feedback in its prompt
   and adjusts accordingly

**What to highlight:**
- Feedback persists across sessions (stored in CSV)
- Only recent feedback (last 10 entries) is injected into the LLM prompt
- Both the original question and the rating are included so the LLM knows
  which topics need better answers

---

## 6. Clear Chat Commands

We support two ways to reset the conversation:

**Option 1: Sidebar button** - Click the "Clear Chat" button at the bottom of the sidebar.

**Option 2: Inline commands** - Type any of the following directly in the chat input. The app intercepts these before they reach the RAG pipeline and resets `st.session_state.messages` immediately.

| Type any of these…         | Result                            |
|---------------------------|-----------------------------------|
| `clear`                   | Chat history cleared, fresh start |
| `clear history`           | Same                              |
| `clear screen`            | Same                              |
| `reset`                   | Same                              |
| `new chat`                | Same                              |
| `start over`              | Same                              |

**How it works:** In `app.py`, before the user's input is sent to `retrieve_context()`, we check if it matches any phrase in `CLEAR_CHAT_PHRASES` (defined in `config.py`). If it matches, we clear the session state and call `st.rerun()` to refresh the page. The phrase list is case-insensitive and uses exact matching, so normal questions containing the word "clear" (e.g., "what are the clear risks?") are not affected.

---

## 7. Keyword + Embedding Hybrid Search

The bot combines two retrieval strategies. Show the difference:

| Question                                         | Retrieval method highlighted           |
|--------------------------------------------------|----------------------------------------|
| `What are the controls for high risks?`          | **Keyword** -matches "high" risk level directly |
| `How do we prevent unauthorized spending?`       | **Embedding** -semantic match to "Unauthorized Commitment", "Anti-Deficiency Act Violation" |
| `grant fraud`                                    | **Both** -keyword finds "Grant Application Fraud" title, embedding finds related fraud records |

---

## 8. Record Summarization

When many records match (> 15), the bot groups them by title with counts
instead of listing every row individually.

**Demo:** `list all risks` -returns grouped summary rather than 100+ individual records.

---

## 9. Sidebar Navigation

- **EPA logo** - displayed at the top for branding, also used as the browser tab favicon
- **App description** - always visible context for new users
- **Example question buttons** - click to auto-fill and run a sample query
- **View Full Risk & Control Registry** - link to the source CSV on GitHub so users can browse the raw data
- **Clear Chat button** - one-click reset that clears `st.session_state.messages` and reruns the app

---

## Suggested Demo Script (5 minutes)

1. **Start fresh** -show the clean UI with sidebar and example buttons
2. **Click a sidebar example** -"What controls exist for Hazardous Waste Mishandling?"
   → Show high-confidence result, source citations, Best Match format
3. **Type a typo query** -`total farud`
   → Show it correctly matches fraud records despite the misspelling
4. **Ask an aggregate question** -`How many high risks are there?`
   → Show the summary/count format instead of Best Match
5. **Show feedback** -click 👎 on a response, add a comment, submit
6. **Ask a broad question** -`workplace safety concerns`
   → Show moderate confidence, multiple matches under "Other Controls"
7. **Clear the chat** -type `clear` in the input
   → Chat resets instantly
8. **Open source citations** -expand the "Sources from Registry" section
   → Show row numbers and similarity scores for traceability

---

## Edge Cases Worth Demonstrating

| Scenario                        | Question to ask                          | What happens                                         |
|---------------------------------|------------------------------------------|------------------------------------------------------|
| Multiple typos in one query     | `hazadorus wast mishandeling`            | Still finds Hazardous Waste Mishandling               |
| Informal/short phrasing         | `spill risks`                            | Matches Chemical Spill Response Delay                 |
| Partial term                    | `IP dispute`                             | Matches Intellectual Property Dispute                 |
| Completely unrelated question   | `What is the weather today?`             | "No matching information found" with low confidence   |
| Very broad question             | `tell me everything`                     | Returns grouped summary of registry                   |
| Risk level filter               | `list all critical risks`                | Filters to Critical-level records only                |
| Control type question           | `what detective controls exist?`         | Finds records with Detective control type             |
