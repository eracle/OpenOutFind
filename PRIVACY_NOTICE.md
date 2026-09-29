# Privacy Notice — OpenOutreach Contacts Store

_This notice explains how the central contacts store operated at `hub.openoutreach.app` collects, uses, and discloses personal data. It is published for the people whose work contact details may appear in the store and for the operators who contribute to and read from it._

## Who operates the store

The store is operated by the maintainer of the open-source project **OpenOutreach**. It holds two kinds of record pooled across the OpenOutreach operator network: **work email addresses**, so a contact one operator has already resolved can be served to another, lowering each operator's email-finder spend as coverage grows; and **professional profiles** returned by operators' prospect searches, so the pool of business contacts does not depend on any one data provider staying available.

Throughout this notice, an **operator** is a person running a self-hosted instance of OpenOutFind (or another tool in the OpenOutreach family) that contributes to and reads from the store.

## What data is in the store

**Email records.** For a person whose work email an operator resolved, the store holds:

| Field | Example | Why it is kept |
| --- | --- | --- |
| Profile identifier | `https://…/in/jane-doe` | A stored profile URL — held as an opaque key, never fetched — used for contribution and resolution. |
| Country code | `in` | Drives the geographic exclusion below. |
| Work email address(es) | `jane@acme.com` | The contact detail the store exists to serve. |

**Profile records.** For a person who appeared in an operator's prospect search, the store holds:

| Field | Example | Why it is kept |
| --- | --- | --- |
| Profile identifier | `https://…/in/jane-doe` | The same opaque key as above. |
| Country code | `in` | The country the search targeted; drives the geographic exclusion below. |
| Professional fields | name, headline, job title, seniority, industry, state/region and country, employer name, domain and industry | What the data provider returned for the search, kept so a person can be judged against a business's ideal customer without searching the provider again. |

With both kinds of record the store also holds a **384-dimension numeric profile vector** (an "embedding"): a compact mathematical representation of the professional fields, **computed on the operator's own machine**, used to rank people by how closely they resemble a business's ideal customer. It records which operator token sent each record (provenance), which software build sent it, and when.

**What is _not_ collected:** no phone number, postal address, personal email, photo, or free-text description of the employer; no special-category data.

Only **professional, business-context (B2B)** contact data is in scope. Consumer contact details and any special-category data are out of scope and are not collected.

## Geographic exclusion (who is _not_ in the store)

Any person located in the **EU/EEA, the UK, or Switzerland** — or whose location cannot be determined — is **never written to the store**. This exclusion runs authoritatively on the server, at the point data enters the store, regardless of what a contributing client sends.

## How data is collected

Data reaches the store from OpenOutreach operators, at two moments:

- **Email records** arrive **after an operator's paid email-finder lookup returns a verified work email.**
- **Profile records** arrive **whenever an operator's prospect search returns a page of results.** The source of those results is the operator's own account with the B2B data provider **BetterContact** (its Lead Finder search), which compiles publicly available professional information.

The maintainer does **not** scrape any website, buy data, or run searches of its own to populate the store; it is filled only by operators' own searches and lookups, subject to the geographic exclusion above.

## How data is used and disclosed

- **Resolution (disclosure to operators).** An operator may query the store for a person's email before paying a finder service. A match is returned to that operator. This means **an email in the store may be disclosed to operators other than the one who contributed it**, so they can carry out business-to-business outreach. This is a disclosure of personal data to a third party — comparable to commercial B2B contact-data providers. The data is **not sold**.
- **Finding prospects (profile records).** Profile records are used to find people who match a business's description of its customers, for the maintainer's own lead-list service and for operators, without searching the data provider again. Profile records are **not** served to operators through any read endpoint today.
- **No consumer-facing purpose.** The store is not used for advertising to consumers or any consumer-facing purpose.

## Legal basis

Where data-protection law applies, the store relies on **legitimate interest** (Art. 6(1)(f) GDPR and equivalents) for **resolution** (serving a known work email) and for **finding prospects** (profile records): **facilitating business-to-business professional communication using professional contact data**, by disclosing existing professional contacts to operators who carry out their own outreach and by identifying professionals who match a business's customer profile. This does not involve profiling for the store's own marketing, automated decision-making with legal or similarly significant effects, or the **sending** of marketing email by the store — operators send from their own infrastructure and are responsible for their own sends and the anti-spam law that governs them.

A legitimate-interest assessment balances that interest against the rights of the people in the store. The safeguards that keep the balance reasonable are: the **geographic exclusion** (people located in the EU/EEA, UK, or Switzerland — or whose location cannot be determined — are never written to the store, so they are never in the searchable set); **data minimisation** (professional fields only — no phone, address or personal contact detail, and the provider's free-text employer description is dropped before it is sent); a **twelve-month retention limit** on profile records; the **B2B-only, professional-context** scope, with no special-category and no consumer data; and the **objection and suppression** rights below, honoured across the whole store. Operators contributing or resolving data may be controllers or joint controllers and carry their own responsibilities.

## Your rights and how to exercise them

If your work email or professional profile is in the store, you may request **access, correction, or erasure**, and you may **object** to the processing. To exercise any of these, or to be excluded from the store entirely:

- **Suppression / opt-out:** a request submitted to `POST /api/v2/suppress/` (or via the contact route below) removes the record and **blocks the email and public identifier from re-entering** the store. Suppression is honoured across the whole store, including against future re-contribution.

Suppression is recorded immediately as a request; the suppressed identifiers are excluded from the data served to operators, and the underlying records are erased on the store's maintenance cycle.

## Retention

Email records persist while they remain useful for resolution and are refreshed when re-contributed. **Profile records are deleted twelve months after they were received**; a person who appears in a later search is stored again with that later date. A suppressed record is removed from served results immediately and erased from source on the maintenance cycle, and a suppressed profile identifier is refused at entry from then on.

## Contact

Questions, complaints, or data-subject requests: open an issue on the OpenOutreach repository or contact the maintainer at the address published there. If your country has a data-protection regulator (for example the AEPD in Spain, the ICO in the UK, or another EU/EEA supervisory authority), you may also lodge a complaint directly with that regulator.

---

_This notice is published in good faith and may be updated as the store evolves; material changes will be reflected here._
