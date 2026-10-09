# Better Data for Fairer, Safer Lending: One-Page Summary

*A plain-language summary of this project. For the technical details, see the [README](README.md).*

## The question

When a lender decides who gets a personal loan, it has to balance two risks: turning away people who would have repaid, and approving people who won't. Most decisions lean heavily on a credit score. **This project asks whether looking a little deeper into a person's credit history leads to better decisions, especially for borrowers with fair or average credit.**

## What I did

I used the real records of **589,000 personal loans** made by LendingClub between 2012 and 2015, where we know how every loan turned out (paid off or not). I built models from the older loans (2012–2014), then tested them on 2015 loans they had never seen. This is the same situation a lender faces: decide today, find out later.

I compared three ways of ranking applicants from safest to riskiest:
1. **LendingClub's own risk grade**, the benchmark.
2. **A model using only the basics:** credit score, income, debt, and loan size.
3. **A model that also looks deeper into the credit file:** for example, how many new accounts someone opened recently, how close their cards are to the limit, and how long they've had credit.

## What I found

**Looking deeper made the difference, not a fancier algorithm.** The basics-only model did no better than LendingClub's grade. The deeper model did, and its biggest gains came among borrowers with fair or average credit (scores below 700), who make up nearly two-thirds of these loans.

For every 100,000 applicants, the deeper model could:
- **approve about 3,750 more people** while keeping the default rate unchanged, or
- **prevent about 400 defaults** while approving the same number of people.

The strongest warning signs were monthly payment relative to income, credit score, and **opening many new accounts in the past two years**, which a credit score alone partly misses.

## Is it fair?

Lending models must not disadvantage people based on race or where they live. The data doesn't record race, so I used Census data to group loans by the demographics and income of each borrower's area.

- **Approval rates stayed close across areas.** Areas with the highest share of minority residents were approved at 98% of the rate of the lowest-share areas. Lower-income areas were approved at 94%. The common regulatory warning line is 80%.
- **The model didn't overstate risk for any group.** If anything, it slightly *under*estimated defaults in higher-minority and lower-income areas.
- **Every declined applicant can be told why.** The model produces specific reasons, such as "payment too high relative to income," which lenders are legally required to give.

One item to watch: in lower-income areas, the new model approves slightly fewer people than LendingClub's grade did (94% vs. 95% of the reference rate). It's well within guidelines, but it's the first thing I would monitor.

## What it can't tell us yet

- **Only approved loans are in the data.** We can't see how rejected applicants would have done.
- **Conditions change.** Defaults rose in 2015, and the model's risk estimates ran low (13.6% predicted vs. 14.9% actual). Adjusting on the most recent year helped only a little. A real lender would need to monitor the model and update it regularly.
- **Profit figures are simplified.** They leave out funding and servicing costs.

## Bottom line

**A careful look beyond the credit score can let a lender say "yes" to more people at the same risk, without creating unfair outcomes,** provided it keeps checking that both stay true over time.
