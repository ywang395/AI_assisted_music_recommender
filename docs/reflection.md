# Reflection

These profile comparisons helped me understand what my recommender is really testing. The system looks at a few weighted features, so the output changes depending on which preferences it can recognize and which ones it cannot.

- `no_contradiction_penalty` vs `Unknown_word`: The contradiction profile still returned songs like `Gym Hero` because the system could still match pop and high energy. The unknown-word profile changed more because the recommender did not understand the genre or mood words, so it had to rely mostly on the numeric values.

- `no_contradiction_penalty` vs `OutOfRangeNumericalValues`: Both profiles gave strange results, but for different reasons. In the contradiction profile, `Gym Hero` keeps showing up because pop and energy are rewarded strongly even when the mood does not fit. In the out-of-range profile, the number targets are unrealistic, so the scoring becomes less meaningful and the system falls back on whatever songs seem closest.

- `OutOfRangeNumericalValues` vs `Unknown_word`: These two profiles both produced less personal recommendations. The out-of-range profile confused the numeric matching, while the unknown-word profile removed the meaning of genre and mood, so both ended up pushing the system toward more generic songs.

- `Happy Pop` vs `Gym Hero`: A song like `Gym Hero` can keep appearing for people who want "Happy Pop" because the recommender gives a lot of weight to pop and high energy. Even if the mood is `intense` instead of `happy`, the strong score from genre and energy can still push it near the top.
