# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

- **VibeFinder 1.0**

---

## 2. Intended Use

- Suggests a small list of songs based on user preferences.
- Uses genre, mood, energy, danceability, valence, acoustic preference, and optional favorite artist.
- Made for classroom exploration, not for real users.

---

## 3. How the Model Works

- Each song is described by genre, mood, energy, danceability, valence, acousticness, and artist.
- The user profile stores the same types of preferences.
- The recommender compares the song to the user and gives a weighted score.
- Exact genre and mood matches matter most.
- Numeric features are scored by how close they are to the user's target values.
- I added small similarity maps for related genres and moods and an optional artist bonus.

---

## 4. Data

- Uses 20 songs from `data/songs.csv`.
- Includes genres like pop, lofi, rock, ambient, classical, hip-hop, and more.
- Includes moods like happy, chill, intense, peaceful, sad, romantic, and motivated.
- The dataset is small, so many music tastes are missing.

---

## 5. Strengths

- Works best when the user's genre and mood are clearly supported by the dataset.
- Does a good job when energy and acoustic preference are also clear.
- The explanation text helps show why a song was recommended.

---

## 6. Limitations and Bias

- Unsupported genre or mood words can be ignored by the system.
- In those cases, the recommender relies more on numeric features like energy and valence.
- This can make different users get similar results.
- Users with unusual or mixed preferences are not represented as well.

---

## 7. Evaluation

- Tested several user profiles and checked whether the top songs matched the preferences.
- Tried normal profiles and edge cases like conflicting preferences, unknown words, and out-of-range numbers.
- Looked at both the ranking order and the explanation text.
- Found that unusual inputs often made the system rely too much on numeric features.

---

## 8. Future Work

- Add input validation for unsupported words and invalid numbers.
- Improve the similarity map for genres and moods.
- Add more diversity so the same type of songs do not always appear at the top.
- Expand the dataset and include more features like tempo or lyrics.

---

## 9. Personal Reflection

- My biggest learning moment was seeing how a simple weighted score could still produce results that looked believable, even when the system was missing important parts of the user's intent.
- AI tools helped me brainstorm test cases, spot weak points in the scoring logic, and explain patterns faster, but I still had to double-check the results by reading the code and comparing them to the actual recommendations.
- What surprised me most was that a basic algorithm could still "feel" smart because repeated patterns like genre and energy matches can make the output seem personal.
- If I extended this project, I would add better input validation, improve the similarity rules, and test ways to make the recommendations more diverse.
