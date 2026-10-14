# Requirements: Free (Sandbox/Test) vs Paid (Production)

To run this project against Google, you need accounts, credentials and product data.
Google offers **test/sandbox environments** that let you validate the full integration
for free. Paid/production is only required to serve real ads and measure real
performance.

## Requirements table

| Requirement | FREE (test / sandbox) | PAID (production) |
|---|---|---|
| Google Merchant Center account | Test/sandbox account, isolated from real data; does not show products to users | Real verified account (domain + business data) |
| Google Ads account | Test account — appears as "cancelled" in the UI, **does not render ads** nor charge | Active account with **billing / card** configured |
| Google Ads API access | Test Account Access: 15,000 operations limit; signed up directly in Google Cloud Console (no classic developer token since Sept 2026) | Basic (15,000 ops) or Standard (unlimited), via request |
| Credentials / Auth | Service account recommended (avoids manual OAuth tokens) | OAuth2 or service account + Customer ID + Merchant ID |
| Google Cloud project | Required (free) to enable the APIs | Required |
| Product data (feed) | Fictitious data OK (invented prices, GTINs). Nothing is published | **Real data mandatory**: valid GTIN, real price, active link, stock |
| Ads visible to public | No — not served; you only validate the code runs | Yes — served and competing in auctions |
| Click / impression cost | $0 — no real auction | Real CPC / conversion spend |
| What you can validate | Code uploads feed, creates campaigns, reads statuses and (simulated) metrics | Real positioning, real metrics (CTR, ROAS, conversions) |

## Key notes

- **Full free development**: Merchant Center test account + Google Ads test account +
  a Google Cloud project is enough to validate the entire flow
  (enrich CSV → upload feed → create campaign → read status) with zero spend and no
  public ads.
- **Free-mode limitation**: test accounts **do not render real ads nor return real
  performance metrics**. They validate that the integration works, not real
  positioning. The optimization loop (Milestone 4) can only be fully tested in
  production with billing.
- **Sept 2026 change**: the classic Google Ads developer token is no longer required;
  access is managed from Google Cloud Console. Simplifies onboarding.
- **Deprecation warning**: the old *Content API for Shopping* is being discontinued
  (Aug 2026). Target the new **Merchant API**, which has its own sandbox test accounts.

## Recommended free setup order

1. Create a project in Google Cloud Console.
2. Enable **Merchant API** and **Google Ads API**.
3. Create a **test account** in Merchant Center.
4. Create a **test account** in Google Ads.
5. Configure a **service account** and download its JSON key.
6. Store credentials outside the repo (env vars / secret manager) — never commit keys.

## Sources

- [Google Ads API — test accounts](https://developers.google.com/google-ads/api/docs/best-practices/test-accounts)
- [Google Ads API — access levels](https://developers.google.com/google-ads/api/docs/api-policy/access-levels)
- [Merchant API — test accounts](https://developers.google.com/merchant/api/guides/accounts/test-accounts)
- [New onboarding without developer token](https://ads-developers.googleblog.com/2026/09/new-onboarding-experience-for-google-ads-api.html)

> Content was rephrased for compliance with licensing restrictions. Verify current
> pricing and policies against the official Google documentation before production use.
