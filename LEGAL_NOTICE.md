# LEGAL NOTICE – OpenOutFind

**Effective upon use of this software**

OpenOutFind is a self-hosted, open-source **lead finder**. It discovers B2B leads from a **licensed third-party data provider**, qualifies them against your described ICP on your own machine, optionally resolves a work email for the best-fit leads through a **paid third-party email-finder**, and hands the result to you as a file. It is **browserless: it does not use, log into, scrape, or automate any social network or professional-network account, and it stores no such credentials.** By running this software, you acknowledge and accept the following facts, risks, and terms.

> **It no longer sends email.** Earlier versions ran cold outreach from mailboxes you owned, with an unsubscribe mechanism, a promotional campaign of the maintainer's own, and an attribution line on every message. All of it has been removed, and the obligations that came with it now sit with whatever tool you send with — see Sections 4 and 5.

> This notice describes how the software behaves and is **not legal advice**. You are responsible for your own compliance; where the stakes warrant it, consult a lawyer. Material aspects of the data model below are still pending a formal legal review.

### 1. No Platform Scraping or Automation
OpenOutFind performs **no** automated access to any social or professional network — no login, no browser session, no bot, no scraping, no messaging on such a platform. Lead **discovery** comes from a licensed data provider (currently BetterContact **Lead Finder**), and **enrichment** (resolving a work email) comes from a paid email-finder — both third-party services **you** sign up for and configure with **your own** API key, used under **that provider's** terms.

- **Profile URLs are identifiers, not fetch targets.** A discovered lead may carry a professional-network profile URL as an opaque identifier. OpenOutFind **stores it and never visits it** — it is a lookup/dedup key, nothing more.
- **You accept the third-party terms.** You are responsible for using the data provider and email-finder in line with each provider's terms of service and acceptable-use policy.

### 2. Newsletter Subscription (Asked at Onboarding, Default Set by Jurisdiction)
During onboarding you enter the **country** your operation is based in, and you are then **asked** whether to subscribe the email address you provided to the newsletter. The question is always asked; only its **default answer** depends on your jurisdiction.

- **Protected jurisdictions**: for operators based in the EU/EEA, UK, Switzerland, Canada, Brazil, Australia, Japan, South Korea, or New Zealand, the default is **no**. An explicit yes is lawful consent anywhere.
- **Elsewhere**: the default is **yes** — so accepting the prompt without changing it subscribes you.
- **Unknown location**: if the country cannot be read, the software treats you as protected (default no).
- **Opting out later**: the choice is made once, at onboarding, and is acted on immediately (a single subscription request); **there is no stored setting to change afterwards**. To leave the list later, use the unsubscribe link in any newsletter email.

### 3. No Warranty – Use at Your Own Risk
OpenOutFind is provided **AS IS**, without warranties of any kind (express or implied), including fitness for a particular purpose, non-infringement, or that it will not cause harm to your accounts, mailboxes, domains, or data.

The developer(s):
- Do not guarantee any results from using the tool
- Are not responsible for account/domain/mailbox suspensions, deliverability harm, lost business, legal consequences, or other damages
- Recommend you review the terms of every third-party service you connect (data provider, email-finder) before use

### 4. How the Project Is Funded (Affiliate Links)
OpenOutFind is free and open-source. It sustains itself through **affiliate links**: the unavoidably-paid third-party service the tool relies on — the email-finder, which powers both lead discovery and address resolution — is surfaced during onboarding through an affiliate link. If you sign up through one, the project may earn a commission **at no markup to you**. You are free to sign up any other way.

OpenOutFind never sends a message and never touches your mailbox, so there is no promotional campaign of the project's own and no attribution line appended to anything. Any hosted service operated by the maintainer is **not** covered by this notice and states its own terms at sign-up.

### 5. Lead Discovery and Email Enrichment
**OpenOutFind does not send email.** It discovers leads, judges them against your described ICP, and writes the result to a file you export. Whatever you use to contact the people it finds is a separate tool you choose, configure and are responsible for.

Address resolution runs through a **third-party email-finder** (e.g. BetterContact) — a paid service you sign up for and configure yourself. It is optional: leads export with their qualification reason whether or not an address was resolved.

- **Data protection**: resolving and storing a person's work email is processing of personal data. Where data-protection law applies (GDPR, UK GDPR, LGPD, etc.) **you are the data controller** and are responsible for a lawful basis, honouring access/erasure/objection requests, and any required disclosures. OpenOutFind provides the mechanism, not legal cover.
- **Anti-spam law is entirely on the tool you send with.** Unsolicited commercial email is regulated — CAN-SPAM (US), GDPR/ePrivacy (EU/EEA), CASL (Canada), the Spam Act (Australia), and others. Requirements commonly include truthful sender and subject lines, a valid physical postal address, and a working, honoured opt-out. **None of these are provided by this software**, because it emits no message in which to provide them.
- **The opt-out duty belongs to whoever sends.** OpenOutFind has no opt-out mechanism of its own, because it never contacts anyone. Your sequencer is the only thing that can honour an opt-out, because it is the only thing that makes contact. Instantly and Smartlead both block a suppressed address at import; confirm your own tool does the same before your first send.
- **Turn on your sequencer's import deduplication.** Exporting the same lead twice can otherwise contact the same person twice. It is opt-in on Smartlead and undocumented on Instantly. This is the one duty the split hands to you that the software cannot do for you.
- **Accuracy**: finder results may be wrong, stale, or belong to a different person. You are responsible for whom you contact and what you send.

### 6. Central Contacts Store (Contribution and Resolution)
OpenOutFind connects to an optional **central contacts store operated by the project maintainer** (`hub.openoutreach.app`). It pools work email addresses across the OpenOutreach network so a contact one operator has already paid to resolve can be served — for free — to another, lowering everyone's email-finder spend as coverage grows. By running the software with contribution enabled you participate as described here.

- **What is contributed, and when**: at the **one** moment a real contact comes into existence — **after a paid email-finder returns a verified work email** — OpenOutFind sends a minimal record: the person's **profile identifier** (the stored, never-fetched profile URL), their **country code**, and the **work email address(es)** resolved. No name, headline, company, title, phone, or profile text is sent. Where a vector for that person is already cached on your machine, the record also carries a **384-dimension numeric profile vector** computed locally — the raw profile text never leaves your machine. (There is no separate switch for the vector: it is included whenever it is already in hand.)
- **Whether you contribute is derived from your country — it is not a setting.** If your operation is **not** based in the EU/EEA, UK, or Switzerland, contribution is **on**, and **there is no toggle to turn it off**: it can be disabled only by modifying the source, which the licence permits. If your operation **is** based there, the software contributes nothing at all (an unreadable country is treated as protected).
- **The consequence for protected operators.** The store works give-to-get: an operator's access token is minted by their **first contribution**. An EEA/UK/CH-based operator therefore never contributes, never earns a token, and so **never resolves from the store** — every lookup falls through to the paid finder. This is a structural consequence of the jurisdiction rule, not a penalty, and it means the store cannot lower your costs if you are based there.
- **Geo-gate on the people in the store**: independently of where *you* are, a contact located in the **EU/EEA, UK, or Switzerland — or whose location cannot be determined — is never written to the store.** This gate runs authoritatively **server-side**; the client's pre-filter is only a bandwidth optimisation.
- **Resolution is a disclosure to third parties.** OpenOutFind reads the store *first*, before spending a paid finder credit. A hit is served free. So an email you contribute **may be disclosed to other operators** to contact that person, and emails others contributed may be disclosed to you. This is a disclosure of personal data to a third party — in substance the commercial-contact-data model (Apollo, Cognism, Dropcontact). It is **not** a sale of data, but it **is** a separate processing purpose from your own outreach.
- **Your role and responsibilities.** Where data-protection law applies, contributing and resolving personal data is processing for which you may be a controller or joint controller alongside the maintainer. **You remain responsible** for a lawful basis (the project relies on legitimate interest for B2B professional contact data only), for honouring access/erasure/objection requests, and for any required notices.
- **Suppression / opt-out.** Any person whose email is in the store can be removed and blocked from re-entry via the store's suppression mechanism (`POST /api/v2/suppress/`), honoured across the whole store. The store publishes a separate **Privacy Notice** for those people at <https://hub.openoutreach.app/privacy/>.

### 7. Your Responsibility
By downloading, installing, configuring, or running OpenOutFind, you:
- Confirm you are of legal age and have authority to accept these terms
- Agree to use the tool only in compliance with all applicable laws (data-protection/privacy law such as GDPR, anti-spam law such as CAN-SPAM/CASL) and with the terms of every third-party service you connect
- Accept full responsibility for the contacts you process, and for any email you go on to send to them with another tool
- Understand that modifying the code to disable the hub contribution is permitted under the licence, but remains your responsibility

If you do **not** agree with any part of this notice — especially the central contacts store — **do not use this software**. Delete it immediately.

Questions or concerns? Open an issue on the repository or contact the maintainer(s).

**Continued use constitutes acceptance of this Legal Notice.**
