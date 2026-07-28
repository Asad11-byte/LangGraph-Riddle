import { useState } from "react";
import { startGame, submitGuess } from "./api";

const DIFFICULTIES = ["easy", "medium", "hard"];

export default function App() {
  // --- setup form state ---------------------------------------------------
  const [playerId, setPlayerId] = useState("player-1");
  const [difficulty, setDifficulty] = useState("medium");
  const [theme, setTheme] = useState("general");
  const [maxAttempts, setMaxAttempts] = useState(3);

  // --- active game state ----------------------------------------------------
  const [threadId, setThreadId] = useState(null);
  const [riddle, setRiddle] = useState(null);
  const [hint, setHint] = useState(null);
  const [failedAttempts, setFailedAttempts] = useState(0);
  const [gameStatus, setGameStatus] = useState("idle"); // idle | waiting_for_user | solved | failed
  const [targetAnswer, setTargetAnswer] = useState(null);
  const [pointsAwarded, setPointsAwarded] = useState(null);
  const [playerScore, setPlayerScore] = useState(null);

  // --- form input for guesses --------------------------------------------------
  const [guessInput, setGuessInput] = useState("");

  // --- ui state ------------------------------------------------------------
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function applyGameResponse(res) {
    setThreadId(res.thread_id);
    setGameStatus(res.game_status);
    setRiddle(res.current_riddle);
    setHint(res.last_hint);
    setFailedAttempts(res.failed_attempts);
    setTargetAnswer(res.target_answer);
    setPointsAwarded(res.points_awarded);
    setPlayerScore(res.player_score);
  }

  async function handleStartGame() {
    setLoading(true);
    setError(null);
    try {
      const res = await startGame({
        playerId,
        difficulty,
        theme,
        maxAttempts: Number(maxAttempts),
      });
      applyGameResponse(res);
      setGuessInput("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function handleSubmitGuess(e) {
    e.preventDefault();
    if (!guessInput.trim() || !threadId) return;
    setLoading(true);
    setError(null);
    try {
      const res = await submitGuess({ threadId, guess: guessInput.trim() });
      applyGameResponse(res);
      setGuessInput("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handlePlayAgain() {
    setThreadId(null);
    setRiddle(null);
    setHint(null);
    setFailedAttempts(0);
    setGameStatus("idle");
    setTargetAnswer(null);
    setPointsAwarded(null);
    setGuessInput("");
    setError(null);
  }

  const gameOver = gameStatus === "solved" || gameStatus === "failed";
  const gameActive = gameStatus === "waiting_for_user" || gameOver;

  return (
    <div className="app">
      <header className="app-header">
        <h1>AI Riddle &amp; Logic Game Master</h1>
        {playerScore !== null && (
          <div className="score-badge">Score: {playerScore}</div>
        )}
      </header>

      {error && <div className="error-banner">{error}</div>}

      {!gameActive && (
        <section className="setup-panel">
          <div className="field">
            <label htmlFor="playerId">Player ID</label>
            <input
              id="playerId"
              type="text"
              value={playerId}
              onChange={(e) => setPlayerId(e.target.value)}
            />
          </div>

          <div className="field">
            <label htmlFor="difficulty">Difficulty</label>
            <select
              id="difficulty"
              value={difficulty}
              onChange={(e) => setDifficulty(e.target.value)}
            >
              {DIFFICULTIES.map((d) => (
                <option key={d} value={d}>
                  {d}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label htmlFor="theme">Theme</label>
            <input
              id="theme"
              type="text"
              value={theme}
              onChange={(e) => setTheme(e.target.value)}
              placeholder="e.g. nature, objects, abstract"
            />
          </div>

          <div className="field">
            <label htmlFor="maxAttempts">Max Attempts</label>
            <input
              id="maxAttempts"
              type="number"
              min={1}
              max={10}
              value={maxAttempts}
              onChange={(e) => setMaxAttempts(e.target.value)}
            />
          </div>

          <button onClick={handleStartGame} disabled={loading || !playerId}>
            {loading ? "Generating riddle..." : "Start Game"}
          </button>
        </section>
      )}

      {gameActive && (
        <section className="game-panel">
          <div className="riddle-card">
            <h2>Riddle</h2>
            <p className="riddle-text">{riddle}</p>
            <p className="attempts-counter">
              Failed attempts: {failedAttempts} / {maxAttempts}
            </p>
            {hint && !gameOver && (
              <p className="hint-text">💡 {hint}</p>
            )}
          </div>

          {!gameOver && (
            <form className="guess-form" onSubmit={handleSubmitGuess}>
              <input
                type="text"
                value={guessInput}
                onChange={(e) => setGuessInput(e.target.value)}
                placeholder="Type your guess..."
                disabled={loading}
                autoFocus
              />
              <button type="submit" disabled={loading || !guessInput.trim()}>
                {loading ? "Checking..." : "Submit Guess"}
              </button>
            </form>
          )}

          {gameOver && (
            <div className={`result-card ${gameStatus}`}>
              <h3>{gameStatus === "solved" ? "🎉 Solved!" : "❌ Out of attempts"}</h3>
              <p>
                The answer was: <strong>{targetAnswer}</strong>
              </p>
              {gameStatus === "solved" && (
                <p>Points awarded: {pointsAwarded}</p>
              )}
              <button onClick={handlePlayAgain}>Play Again</button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}
