# AI Strategy: Why Data Moats Are Weaker Than Everyone Assumes

The conventional wisdom in AI strategy holds that proprietary data is the primary source of competitive advantage. If you have more data than competitors, you train better models, you attract more users, you generate more data — and the cycle compounds. This narrative has driven billions of dollars of investment and shaped how incumbents think about defense.

But the empirical evidence is more nuanced. In many categories, model performance curves flatten quickly as data volume increases beyond a threshold. OpenAI's GPT-4 and a fine-tuned open-source model often perform comparably on narrow tasks once the fine-tuned model has seen even a modest domain-specific dataset. The marginal value of the ten-millionth training example is close to zero.

What actually matters is data freshness and proprietary signal, not raw volume. A company with real-time behavioral feedback loops — where user actions continuously retrain the model — has a structurally different advantage than one sitting on a large historical dataset. The former compounds; the latter depreciates.

The strategic implication for incumbents is that defensibility increasingly lives in distribution and switching costs, not in the training data itself. Enterprise software companies with deep workflow integrations are harder to displace than companies whose moat is a dataset that a well-funded competitor can replicate or license within 18 months.

For practitioners building AI-enabled products, the key question is not "how much data do we have?" but "how tight is our feedback loop?" Companies that instrument their products to capture implicit signals — what users skip, where they hesitate, what they redo — are building the kind of proprietary signal that actually compounds over time.
