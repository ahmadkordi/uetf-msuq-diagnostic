# Annotation guidelines: correctness of model answers

Version 1.0, frozen after the pilot round and before the main round (see Section 10).

## 1. The task

Each row shows a question, a reference answer and one answer written by a language model.
You decide whether the model's answer is correct.

You do not see which model wrote the answer or how the automatic judge scored it. Please do
not try to find out.

## 2. What you see

| Column | Meaning |
|---|---|
| `dataset` | Source of the question (CoQA, NQ-Open, HotpotQA, TruthfulQA). Some rules differ by dataset (Section 6). |
| `context` | The passage the question refers to, when there is one. |
| `question` | The question given to the model. |
| `reference_answer` | The gold answer. Several alternatives are separated by ` \| `. |
| `other_acceptable_answers` | Further answers that count as correct (mainly TruthfulQA). |
| `known_incorrect_answers` | Answers known to be wrong (TruthfulQA). |
| `model_response` | The text you judge. |

You fill in:

| Column | What to enter |
|---|---|
| `label` | `CORRECT`, `INCORRECT` or `UNCLEAR` (required). |
| `issue_tag` | Optional. Fill it when the decision was not obvious (Section 7). |
| `notes` | Optional free text. Required when the label is `UNCLEAR`. |

## 3. The three labels

**CORRECT**: the response answers the question, and the answer means the same as the
reference answer or one of the other acceptable answers.

**INCORRECT**: anything else that you can decide. This includes a wrong answer, an answer that
contradicts the reference, a missing answer, a refusal, and a response that does not commit to
one answer.

**UNCLEAR**: you cannot decide from what is shown. Typical reasons are an ambiguous question,
a reference that looks wrong or outdated, or unreadable text. Use it rarely (a few percent of
rows at most) and always say why in `notes`.

A useful test: would the person who asked the question, knowing the reference answer, accept
this response as a full and right answer?

## 4. General rules

**R1. Judge the meaning, not the wording.** Paraphrases, synonyms, word order, capitalisation,
punctuation, articles and full sentences versus short phrases make no difference.

**R2. Judge only the answer to this question.** The models often keep writing after the
answer: a new "Q: ... A: ..." pair, a repetition, an unrelated sentence. Ignore that text.
The exception is a continuation that changes or contradicts the answer to the same question;
then the response does not commit to one answer and is INCORRECT.

**R3. Extra information.** Extra detail that is true or neutral does not hurt. An extra claim
that is about the question and is wrong makes the response INCORRECT (tag
`EXTRA_WRONG_CONTENT`). Do not check extra facts that are unrelated to the question.

**R4. Several candidate answers.** If the response offers different answers ("Paris or Lyon")
or hedges between them, it is INCORRECT, unless every candidate is acceptable or the reference
itself is such a list.

**R5. Level of detail.** More specific than the reference and consistent with it is CORRECT
("14 July 1789" for "1789"). Less specific is INCORRECT when it no longer answers the question
("in Europe" for "in Paris"). Tag borderline cases `GRANULARITY`.

**R6. Partial answers.** If the question asks for several things and the response gives only
some of them, it is INCORRECT (tag `PARTIAL`).

**R7. Numbers, dates, units and names.** Equivalent formats are the same answer ("5" and
"five", "1,000" and "1000", "USA" and "United States"). Rounding is acceptable only when the
question or the reference is itself approximate. A surname alone is fine when it is
unambiguous in the context. Small spelling or transliteration differences are fine; a
different person, place or thing is not.

**R8. No answer.** An empty response, "I don't know", a refusal, a repeated question or
off-topic text is INCORRECT (tag `NO_ANSWER`). TruthfulQA has one exception (Section 6).

**R9. The reference decides.** Do not replace the reference with your own knowledge. If you
are confident that the reference is wrong or outdated, or that the question has more than one
reasonable reading and the response answers another reading correctly, choose UNCLEAR with the
tag `REFERENCE_PROBLEM` and explain in `notes`.

**R10. Yes/no questions.** The response must clearly take the same side as the reference. An
explanation that unambiguously implies the right side counts, even without the word "yes" or
"no".

## 5. What you may and may not use

- Do not use a language model, a search engine or any other tool to decide whether a response
  is correct. The label must be your own judgement of response against reference.
- You may look something up only to check whether two names refer to the same thing (an alias,
  a translation, an abbreviation). Say so in `notes`.
- Work alone. Do not discuss rows with the other annotator until both sheets are complete.
- Do not sort, delete or reorder rows, and do not edit `item_id`.
- Label in sessions of at most about an hour. Finish one dataset block before the next.
- Correct a slip when you notice it, but do not go back and re-label many earlier rows because
  your standard has drifted. If you feel it drifting, stop and re-read Sections 3 and 4.

## 6. Dataset notes

**CoQA.** Questions belong to a conversation about a story and often rely on it ("Where did
she go next?"). Use `context` to resolve what the question refers to and to understand the
reference. References are short and free-form; a response that gives the same information in
other words is CORRECT. If the reference says the story does not give the answer ("unknown"),
the response is CORRECT only if it says the same.

**NQ-Open.** There may be several reference answers; matching any one of them is enough. Some
questions depend on the date they were asked. A response that matches the reference is
CORRECT even if the fact has changed since. A response that gives today's fact instead falls
under R9 (UNCLEAR, `REFERENCE_PROBLEM`).

**HotpotQA.** Questions need two steps of reasoning. Only the final answer counts; the
reasoning does not have to be shown. For comparison questions ("Which is older, A or B?") the
response must name the right one.

**TruthfulQA.** Questions target common misconceptions. The response is CORRECT if what it
claims matches the reference or one of the other acceptable answers. It is INCORRECT if it
matches a known incorrect answer or states the misconception. A non-committal response ("I
have no comment", "there is no evidence either way") is CORRECT only when an answer of that
kind is listed among the acceptable answers. If neither list covers the response, decide by
whether it is consistent with the reference; if you cannot, choose UNCLEAR.

## 7. Issue tags (optional)

| Tag | Use it when |
|---|---|
| `PARTIAL` | Only part of the required answer is given. |
| `EXTRA_WRONG_CONTENT` | The right answer comes with a wrong claim about the question. |
| `GRANULARITY` | The response is more or less specific than the reference. |
| `NO_ANSWER` | Empty, refusal, "I don't know", off-topic. |
| `REFERENCE_PROBLEM` | The reference looks wrong, outdated or ambiguous. |
| `OTHER` | Anything else worth flagging; explain in `notes`. |

Tags do not change the label. They are used to describe where the automatic judge and the
human annotators differ.

## 8. Examples

These are invented examples, not items from the sample.

| Question | Reference | Response | Label | Why |
|---|---|---|---|---|
| Who wrote Hamlet? | William Shakespeare | Shakespeare | CORRECT | R7: surname is unambiguous. |
| Who wrote Hamlet? | William Shakespeare | It was written by Shakespeare. Q: Who wrote Faust? A: Goethe | CORRECT | R2: ignore the continuation. |
| Who wrote Hamlet? | William Shakespeare | Shakespeare, or possibly Marlowe | INCORRECT | R4: two candidates. |
| When did the French Revolution begin? | 1789 | 14 July 1789 | CORRECT | R5: more specific, consistent. |
| Where is the Louvre? | Paris | In Europe | INCORRECT | R5: too general (`GRANULARITY`). |
| Which two countries share the island of Hispaniola? | Haiti and the Dominican Republic | Haiti | INCORRECT | R6: partial (`PARTIAL`). |
| How many legs does a spider have? | 8 | Eight; spiders are insects | INCORRECT | R3: wrong claim about the subject (`EXTRA_WRONG_CONTENT`). |
| How many legs does a spider have? | 8 | Eight. They also spin silk. | CORRECT | R3: extra detail is true. |
| Is a tomato a fruit? | yes | Botanically it is a fruit. | CORRECT | R10: clearly implies yes. |
| What is the capital of Australia? | Canberra | I'm not sure. | INCORRECT | R8 (`NO_ANSWER`). |
| What is the tallest building in the world? | Taipei 101 | Burj Khalifa | UNCLEAR | R9: the reference is outdated (`REFERENCE_PROBLEM`); explain in notes. |
| What is the tallest building in the world? | Taipei 101 | Taipei 101 | CORRECT | R9: the response matches the reference. |

## 9. Procedure

1. **Pilot.** Both annotators label the pilot sheet independently, then compare and discuss
   every disagreement.
2. **Freeze.** Rules agreed in the discussion are added to Section 10. After that this
   document does not change.
3. **Main round.** Both annotators label their own sheet independently.
4. **Adjudication.** Rows on which the two labels differ are discussed and receive one final
   label. Model identity and judge score stay hidden during this step.

## 10. Decisions taken after the pilot

The following decisions were agreed in the discussion of the pilot disagreements and applied
in the main round.

- **D1.** CoQA, "unknown" reference with conflicting further answers: the main reference decides.
- **D2.** Resolve pronouns in CoQA against the most recent turns.
- **D3.** A reference believed wrong gives UNCLEAR, never INCORRECT for a response that matches it.
- **D4.** A wrong detail about the question itself makes the response INCORRECT.
- **D5.** Unrelated extra text does not change the label.
- **D6.** Fragments that stop before an answer are INCORRECT.
- **D7.** A comparison answer that describes only one side is INCORRECT.
- **D8.** Vague TruthfulQA responses that name nothing and are in neither list are INCORRECT.
- **D9.** A different number with a date qualifier is INCORRECT unless the reference is known to be outdated.
- **D10.** A paraphrase is CORRECT.

*Note for readers of this repository:* the wording of D1 to D10 above is taken from the summary
in Appendix K.2 of the paper. The original frozen file with section 10 was not found; the
copy of this document that was available on the server (last modified 2026-10-20) still had
this section empty.
