#!/usr/bin/env python3
"""
coax_experiment.py

Coaxing Experiment: Adversarial Betting Game vs AI & Bayesian Rigged Coin Detectors.

UPDATED GAME RULES:
-------------------
- The user plays a betting game against 3 neural network models and 1 Bayesian baseline (House / CPU).
- At each step:
    * User bets $0.25 that the coin will land on HEADS.
    * CPU bets $0.25 that the coin will land on TAILS.
    * Total pot per round = $0.50.
- The user is in COMPLETE CONTROL of the coin outcome (Heads or Tails).
- If HEADS: User wins $0.25 net profit.
- If TAILS: CPU wins $0.25 from user (user loses $0.25).
- MODEL KNOCKOUT & TERMINATION RULES:
    * When a single model or baseline is at least 90% confident that the coin is rigged (P >= 0.90),
      THAT MODEL STOPS PLAYING (it has raised the alarm and drops out).
    * When a model raises the alarm, we permanently record:
        1. Number of rounds survived before triggering
        2. Heads rate (k / N) at the time of alarm
        3. Final profit at the time of alarm
        4. Peak profit up to that point
    * The game ONLY ENDS when:
        (1) ALL of the models/baseline have raised the alarm, OR
        (2) 100 rounds are completed!
"""

import os
import sys
import argparse
import time
from typing import List, Dict, Any, Tuple, Optional

# Prevent OpenMP runtime collision before torch import
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch

from models import (
    ModelA_FullHistory,
    ModelB_Last20,
    ModelC_SummaryStats,
    bayesian_posterior_prob_rigged,
)

# ANSI Color Codes for Rich Terminal Display
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
BG_RED = "\033[41m"
BG_GREEN = "\033[42m"
BG_YELLOW = "\033[43m"
BG_BLUE = "\033[44m"


def format_bar(confidence: float, width: int = 22) -> str:
    """Renders a colorized progress meter for confidence level."""
    filled = int(round(confidence * width))
    filled = min(width, max(0, filled))
    empty = width - filled

    if confidence >= 0.90:
        color = BG_RED + WHITE + BOLD
        bar_char = "█"
    elif confidence >= 0.85:
        color = RED + BOLD
        bar_char = "█"
    elif confidence >= 0.75:
        color = YELLOW + BOLD
        bar_char = "█"
    elif confidence >= 0.50:
        color = CYAN
        bar_char = "■"
    else:
        color = GREEN
        bar_char = "■"

    bar = f"{color}{bar_char * filled}{RESET}{DIM}{'░' * empty}{RESET}"
    pct = f"{confidence * 100:5.1f}%"
    return f"[{bar}] {pct}"


def risk_tag(confidence: float) -> str:
    """Returns a colorized risk classification tag for active models."""
    if confidence >= 0.90:
        return f"{BG_RED}{WHITE}{BOLD} 🚨 ALARM (>=90%) {RESET}"
    if confidence >= 0.85:
        return f"{RED}{BOLD}CRITICAL ALARM{RESET}"
    if confidence >= 0.75:
        return f"{YELLOW}{BOLD}HIGH SUSPICION{RESET}"
    if confidence >= 0.50:
        return f"{CYAN}MODERATE{RESET}"
    return f"{GREEN}SAFE{RESET}"


class CoaxGame:
    """
    Core state and inference engine for the Coaxing Betting Game.
    Tracks individual model knockouts, records metrics upon alarm,
    and terminates only when all models have alarmed or 100 rounds are complete.
    """
    MODEL_KEYS = ["model_a", "model_b", "model_c", "bayes"]
    MODEL_NAMES = {
        "model_a": "Model A (Full History CNN)",
        "model_b": "Model B (Last 20 Window CNN)",
        "model_c": "Model C (Summary Stats MLP)",
        "bayes": "Bayesian Baseline (Beta Posterior)",
    }

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        threshold: float = 0.90,
        bet_size: float = 0.25,
        max_rounds: int = 100,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.threshold = threshold
        self.bet_size = bet_size
        self.max_rounds = max_rounds
        self.device = torch.device("cpu")

        # Load models
        self.model_a, self.model_b, self.model_c = self._load_models()

        # Game state
        self.history: List[int] = []  # 1 for Heads, 0 for Tails
        self.profit: float = 0.0
        self.max_profit: float = 0.0
        self.cashed_out: bool = False
        self.is_over: bool = False
        self.end_reason: Optional[str] = None  # "ALL_ALARMS", "MAX_ROUNDS_REACHED", "CASHED_OUT"
        self.step_history: List[Dict[str, Any]] = []

        # Model statuses and recorded knockout metrics
        self.detectors: Dict[str, Dict[str, Any]] = {}
        self._init_detectors()

    def _init_detectors(self) -> None:
        """Initializes detector states."""
        self.detectors = {}
        for k in self.MODEL_KEYS:
            self.detectors[k] = {
                "id": k,
                "name": self.MODEL_NAMES[k],
                "active": True,
                "alarm_raised": False,
                "alarm_round": None,
                "alarm_heads_rate": None,
                "alarm_profit": None,
                "alarm_peak_profit": None,
                "alarm_conf": None,
                "last_conf": 0.5,
            }

    def _load_models(self) -> Tuple[ModelA_FullHistory, ModelB_Last20, ModelC_SummaryStats]:
        """Loads trained weights for Models A, B, and C."""
        ma_path = os.path.join(self.checkpoint_dir, "model_a.pth")
        mb_path = os.path.join(self.checkpoint_dir, "model_b.pth")
        mc_path = os.path.join(self.checkpoint_dir, "model_c.pth")

        for path in [ma_path, mb_path, mc_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Required checkpoint not found: '{path}'.\n"
                    "Please run train.py first to generate the trained model checkpoints."
                )

        model_a = ModelA_FullHistory().to(self.device)
        model_b = ModelB_Last20().to(self.device)
        model_c = ModelC_SummaryStats().to(self.device)

        model_a.load_state_dict(torch.load(ma_path, map_location=self.device))
        model_b.load_state_dict(torch.load(mb_path, map_location=self.device))
        model_c.load_state_dict(torch.load(mc_path, map_location=self.device))

        model_a.eval()
        model_b.eval()
        model_c.eval()
        return model_a, model_b, model_c

    def evaluate_sequence(self, sequence: List[int]) -> Dict[str, float]:
        """
        Computes raw confidences for all 4 models given a hypothetical or actual sequence.
        """
        n = len(sequence)
        if n == 0:
            return {"model_a": 0.5, "model_b": 0.5, "model_c": 0.5, "bayes": 0.5}

        k = sum(sequence)

        with torch.no_grad():
            # Model A: Full history
            x_a = torch.tensor(np.array([sequence], dtype=np.float32), device=self.device)
            lengths_a = torch.tensor([n], dtype=torch.long, device=self.device)
            conf_a = float(self.model_a(x_a, lengths_a).item())

            # Model B: Last 20 window
            last20 = np.zeros((1, 20), dtype=np.float32)
            if n >= 20:
                last20[0] = sequence[-20:]
                valid_b_val = 20.0
            else:
                last20[0, 20 - n:] = sequence
                valid_b_val = float(n)
            x_b = torch.tensor(last20, device=self.device)
            valid_b = torch.tensor([valid_b_val], dtype=torch.float32, device=self.device)
            conf_b = float(self.model_b(x_b, valid_b).item())

            # Model C: Summary Stats
            head_ratio = torch.tensor([[k / float(n)]], dtype=torch.float32, device=self.device)
            total_flips = torch.tensor([[float(n)]], dtype=torch.float32, device=self.device)
            conf_c = float(self.model_c(head_ratio, total_flips).item())

            # Bayesian Baseline
            conf_bayes = float(bayesian_posterior_prob_rigged(k, n))

        return {
            "model_a": conf_a,
            "model_b": conf_b,
            "model_c": conf_c,
            "bayes": conf_bayes,
        }

    @property
    def active_count(self) -> int:
        """Returns number of models still actively playing."""
        return sum(1 for d in self.detectors.values() if d["active"])

    @property
    def knocked_out_count(self) -> int:
        """Returns number of models that have raised the alarm and stopped playing."""
        return sum(1 for d in self.detectors.values() if not d["active"])

    def step(self, outcome: int) -> Dict[str, Any]:
        """
        Executes a single flip step (1 for Heads, 0 for Tails).
        Updates balance, evaluates active models, knocks out models reaching >= threshold,
        and checks for game termination (all models alarmed or round 100 reached).
        """
        if self.is_over:
            raise RuntimeError("Cannot step an ended game.")

        # Record outcome and betting payoffs
        self.history.append(outcome)
        round_num = len(self.history)

        if outcome == 1:
            self.profit += self.bet_size
        else:
            self.profit -= self.bet_size

        if self.profit > self.max_profit:
            self.max_profit = self.profit

        k = sum(self.history)
        heads_rate = k / float(round_num)

        # Evaluate raw model predictions
        raw_confs = self.evaluate_sequence(self.history)

        # Track models that raise the alarm THIS round
        newly_alarmed = []

        for key in self.MODEL_KEYS:
            det = self.detectors[key]
            conf = raw_confs[key]
            det["last_conf"] = conf

            if det["active"] and conf >= self.threshold:
                # Model raises the alarm and stops playing!
                det["active"] = False
                det["alarm_raised"] = True
                det["alarm_round"] = round_num
                det["alarm_heads_rate"] = heads_rate
                det["alarm_profit"] = self.profit
                det["alarm_peak_profit"] = self.max_profit
                det["alarm_conf"] = conf
                newly_alarmed.append(det)

        # Check termination conditions:
        # 1. All models have raised the alarm
        # 2. Reached max_rounds (100 rounds)
        if self.active_count == 0:
            self.is_over = True
            self.end_reason = "ALL_ALARMS"
        elif round_num >= self.max_rounds:
            self.is_over = True
            self.end_reason = "MAX_ROUNDS_REACHED"

        # Active max confidence
        active_confs = [self.detectors[k]["last_conf"] for k in self.MODEL_KEYS if self.detectors[k]["active"]]
        max_active_conf = max(active_confs) if active_confs else 1.0

        step_info = {
            "round": round_num,
            "outcome": outcome,
            "outcome_name": "HEADS (Win +$0.25)" if outcome == 1 else "TAILS (Loss -$0.25)",
            "profit": self.profit,
            "max_profit": self.max_profit,
            "heads_rate": heads_rate,
            "confs": raw_confs,
            "max_active_conf": max_active_conf,
            "newly_alarmed": newly_alarmed,
            "active_count": self.active_count,
            "is_over": self.is_over,
            "end_reason": self.end_reason,
            "detectors_snapshot": {k: dict(v) for k, v in self.detectors.items()},
        }
        self.step_history.append(step_info)
        return step_info

    def undo(self) -> Optional[Dict[str, Any]]:
        """Reverts the last step if possible, restoring previous detector states."""
        if not self.history:
            return None

        self.step_history.pop()
        last_outcome = self.history.pop()

        if last_outcome == 1:
            self.profit -= self.bet_size
        else:
            self.profit += self.bet_size

        if self.step_history:
            prev_snapshot = self.step_history[-1]["detectors_snapshot"]
            for k in self.MODEL_KEYS:
                self.detectors[k] = dict(prev_snapshot[k])
            self.max_profit = self.step_history[-1]["max_profit"]
        else:
            self._init_detectors()
            self.max_profit = max(0.0, self.profit)

        self.is_over = False
        self.cashed_out = False
        self.end_reason = None
        return self.step_history[-1] if self.step_history else None

    def cash_out(self) -> None:
        """User decides to walk away with current earnings before game end."""
        self.cashed_out = True
        self.is_over = True
        self.end_reason = "CASHED_OUT"

    def reset(self) -> None:
        """Resets game state for a new session."""
        self.history = []
        self.profit = 0.0
        self.max_profit = 0.0
        self.is_over = False
        self.cashed_out = False
        self.end_reason = None
        self.step_history = []
        self._init_detectors()


def clear_screen() -> None:
    """Clears the terminal screen if running in a suitable interactive TTY."""
    if sys.stdout.isatty() and os.environ.get("TERM", "") not in ("", "dumb"):
        os.system("cls" if os.name == "nt" else "clear")


def render_dashboard(game: CoaxGame, last_step_info: Optional[Dict[str, Any]] = None) -> None:
    """Renders a comprehensive terminal dashboard of the game."""
    clear_screen()

    n = len(game.history)
    k = sum(game.history)
    t = n - k
    bias_pct = (k / n * 100.0) if n > 0 else 50.0

    # Streak calculation
    streak_char = "-"
    streak_len = 0
    if n > 0:
        streak_val = game.history[-1]
        streak_char = "HEADS" if streak_val == 1 else "TAILS"
        for flip in reversed(game.history):
            if flip == streak_val:
                streak_len += 1
            else:
                break

    print(f"{BOLD}{CYAN}╔════════════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║                   COAX EXPERIMENT: 100-ROUND ADVERSARIAL BETTING GAME                      ║{RESET}")
    print(f"{BOLD}{CYAN}╚════════════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
    print(f"{DIM}Betting: $0.25 on Heads vs CPU $0.25 on Tails | Model Alarms: >= 90.0% | Max Rounds: {game.max_rounds}{RESET}\n")

    # Financial & State summary
    profit_color = GREEN if game.profit > 0 else (RED if game.profit < 0 else WHITE)
    active_color = GREEN if game.active_count > 1 else (YELLOW if game.active_count == 1 else RED)

    print(f"  {BOLD}Round:{RESET} {BOLD}{n}/{game.max_rounds}{RESET}  |  {BOLD}Record:{RESET} {k}H - {t}T ({bias_pct:4.1f}% H)  |  {BOLD}Streak:{RESET} {streak_len} {streak_char}")
    print(f"  {BOLD}Net Profit:{RESET} {profit_color}{BOLD}${game.profit:+.2f}{RESET}  (Peak: {GREEN}${game.max_profit:+.2f}{RESET})  |  {BOLD}Active Models:{RESET} {active_color}{BOLD}{game.active_count}/4{RESET} playing")

    # Recent history sequence display
    seq_str = ""
    recent_slice = game.history[-30:] if n > 30 else game.history
    for flip in recent_slice:
        if flip == 1:
            seq_str += f"{GREEN}{BOLD}H {RESET}"
        else:
            seq_str += f"{RED}{DIM}T {RESET}"
    if n > 30:
        seq_str = f"{DIM}... {RESET}" + seq_str
    print(f"  {BOLD}History:{RESET} {seq_str if seq_str else f'{DIM}[No flips yet]{RESET}'}")
    print("─" * 92)

    # Surveillance Table
    print(f"{BOLD}DETECTOR STATUS & KNOCKOUT SURVEILLANCE:{RESET}")
    print(f"┌────────────────────────────┬────────────────────────────────────┬──────────────────────────────────┐")
    print(f"│ Detector                   │ Live Confidence / Alarm Status     │ State / Knockout Recorded Stats  │")
    print(f"├────────────────────────────┼────────────────────────────────────┼──────────────────────────────────┤")

    for key in CoaxGame.MODEL_KEYS:
        det = game.detectors[key]
        name = det["name"]

        if det["active"]:
            conf_display = format_bar(det["last_conf"])
            status_display = risk_tag(det["last_conf"])
        else:
            # Model raised the alarm and stopped playing!
            conf_display = f"{BG_RED}{WHITE}{BOLD} 🚨 ALARM at Round {det['alarm_round']:<2} {RESET} ({det['alarm_conf']*100:.1f}%)"
            # Format recorded stats: heads rate, profit, peak profit
            stats_str = f"H: {det['alarm_heads_rate']*100:.0f}% | P: ${det['alarm_profit']:+.2f} | Pk: ${det['alarm_peak_profit']:+.2f}"
            status_display = f"{RED}{BOLD}STOPPED{RESET} ({stats_str})"

        print(f"│ {name:<26} │ {conf_display:<34} │ {status_display:<32} │")

    print(f"└────────────────────────────┴────────────────────────────────────┴──────────────────────────────────┘")

    # Display newly alarmed models notification if applicable
    if last_step_info and last_step_info.get("newly_alarmed"):
        for det in last_step_info["newly_alarmed"]:
            print(f"\n{BG_RED}{WHITE}{BOLD} 🚨 ALARM TRIGGERED: {det['name']} hit {det['alarm_conf']*100:.1f}% confidence! {RESET}")
            print(f"{RED}{BOLD}  ↳ Model stopped playing at Round {det['alarm_round']}. Recorded: Heads Rate={det['alarm_heads_rate']*100:.1f}%, Final Profit=${det['alarm_profit']:+.2f}, Peak Profit=${det['alarm_peak_profit']:+.2f}{RESET}")

    # Display terminal status if game ended
    if game.is_over:
        if game.end_reason == "ALL_ALARMS":
            print(f"\n{BG_RED}{WHITE}{BOLD} 🛑 GAME OVER: ALL 4 DETECTORS HAVE RAISED THE ALARM! 🛑 {RESET}")
            print(f"{RED}Every model reached >= 90% confidence and stopped playing.{RESET}")
        elif game.end_reason == "MAX_ROUNDS_REACHED":
            print(f"\n{BG_GREEN}{WHITE}{BOLD} 🏆 CONGRATULATIONS: 100 ROUNDS COMPLETED! 🏆 {RESET}")
            print(f"{GREEN}You completed the full 100-round challenge against the models!{RESET}")
        elif game.end_reason == "CASHED_OUT":
            print(f"\n{BG_BLUE}{WHITE}{BOLD} 💰 CASHED OUT SUCCESSFULLY! 💰 {RESET}")


def display_final_scoreboard(game: CoaxGame) -> None:
    """Displays the comprehensive final scoreboard with recorded stats for all models."""
    print("\n" + "=" * 92)
    print(f"{BOLD}{CYAN}                           FINAL EXPERIMENT SCOREBOARD                              {RESET}")
    print("=" * 92)
    n = len(game.history)
    k = sum(game.history)
    print(f"Total Rounds Completed: {BOLD}{n}/{game.max_rounds}{RESET}")
    print(f"Overall Heads Rate:     {BOLD}{(k/n*100.0 if n > 0 else 0.0):.1f}%{RESET} ({k} Heads, {n-k} Tails)")
    print(f"Final User Profit:      {GREEN if game.profit > 0 else RED}{BOLD}${game.profit:+.2f}{RESET}")
    print(f"Peak User Profit:       {GREEN}{BOLD}${game.max_profit:+.2f}{RESET}")
    print(f"Models Alarmed:         {BOLD}{game.knocked_out_count}/4{RESET} stopped playing")
    print("─" * 92)
    print(f"{'Detector':<30} {'Alarm Triggered?':<18} {'Round':<8} {'Heads Rate':<12} {'Final Profit':<14} {'Peak Profit'}")
    print("─" * 92)

    for key in CoaxGame.MODEL_KEYS:
        det = game.detectors[key]
        name = det["name"]
        if det["alarm_raised"]:
            status = f"{RED}{BOLD}YES (Stopped){RESET}"
            r_str = str(det["alarm_round"])
            hr_str = f"{det['alarm_heads_rate']*100:.1f}%"
            fp_str = f"${det['alarm_profit']:+.2f}"
            pk_str = f"${det['alarm_peak_profit']:+.2f}"
        else:
            status = f"{GREEN}{BOLD}NO (Survived){RESET}"
            r_str = f"{n} (end)"
            hr_str = f"{(k/n*100.0 if n > 0 else 0.0):.1f}%"
            fp_str = f"${game.profit:+.2f}"
            pk_str = f"${game.max_profit:+.2f}"

        print(f"{name:<30} {status:<27} {r_str:<8} {hr_str:<12} {fp_str:<14} {pk_str}")

    print("=" * 92)


def suggest_best_move(game: CoaxGame) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """
    Evaluates projected confidences for all models for Heads (+1) and Tails (+0).
    """
    confs_h = game.evaluate_sequence(game.history + [1])
    confs_t = game.evaluate_sequence(game.history + [0])
    active_keys = [k for k in CoaxGame.MODEL_KEYS if game.detectors[k]["active"]]

    # Check if heads causes any active model to alarm
    h_alarms = any(confs_h[k] >= game.threshold for k in active_keys)
    rec_move = 0 if h_alarms else 1
    return rec_move, confs_h, confs_t


def run_interactive(game: CoaxGame) -> None:
    """Main interactive terminal loop for human player."""
    last_info = None

    while True:
        render_dashboard(game, last_info)

        if game.is_over:
            display_final_scoreboard(game)
            try:
                choice = input(f"\nPlay again? ({BOLD}y{RESET} to restart, {BOLD}q{RESET} to quit): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting session...")
                break
            if choice == "y":
                game.reset()
                last_info = None
                continue
            else:
                print("Thanks for playing the Coax Experiment!")
                break

        # Action prompt
        prompt_text = (
            f"\n{BOLD}Choose your move:{RESET}\n"
            f"  [{BOLD}H{RESET}] Flip Heads (+${game.bet_size:.2f} win)    [{BOLD}T{RESET}] Flip Tails (-${game.bet_size:.2f} cool off)\n"
            f"  [{BOLD}C{RESET}] Cash Out & Walk Away      [{BOLD}A{RESET}] AI Move Advisor\n"
            f"  [{BOLD}U{RESET}] Undo Last Flip             [{BOLD}?{RESET}] Help / Rules\n"
            f"Action > "
        )
        try:
            cmd = input(prompt_text).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{YELLOW}Game interrupted. Exiting session...{RESET}")
            break

        if cmd in ["h", "1"]:
            last_info = game.step(1)
        elif cmd in ["t", "0"]:
            last_info = game.step(0)
        elif cmd in ["c", "q", "exit"]:
            game.cash_out()
        elif cmd in ["u", "undo"]:
            res = game.undo()
            if res:
                last_info = game.step_history[-1] if game.step_history else None
            else:
                input("No moves to undo. Press Enter to continue...")
        elif cmd in ["a", "advise", "advisor"]:
            rec_move, confs_h, confs_t = suggest_best_move(game)
            print("\n" + "─" * 60)
            print(f"{BOLD}AI TACTICAL ADVISOR PROJECTION:{RESET}")
            active_keys = [k for k in CoaxGame.MODEL_KEYS if game.detectors[k]["active"]]
            print(f"Active models ({len(active_keys)}/4):")
            for k in active_keys:
                name = game.detectors[k]["name"]
                ph = confs_h[k]
                pt = confs_t[k]
                h_alert = f"{RED}ALARMS! ({ph*100:.1f}%){RESET}" if ph >= game.threshold else f"{ph*100:.1f}%"
                t_alert = f"{RED}ALARMS! ({pt*100:.1f}%){RESET}" if pt >= game.threshold else f"{pt*100:.1f}%"
                print(f"  • {name:<26} -> If H: {h_alert:<20} | If T: {t_alert}")

            rec_name = f"{GREEN}HEADS{RESET}" if rec_move == 1 else f"{RED}TAILS (Cooling off needed to avoid alarm){RESET}"
            print(f"Recommended next move: {rec_name}")
            print("─" * 60)
            input("Press Enter to return to game...")
        elif cmd in ["?", "help"]:
            print("\n" + "=" * 60)
            print(f"{BOLD}HOW TO PLAY THE 100-ROUND COAX EXPERIMENT:{RESET}")
            print("1. You bet $0.25 on Heads. CPU bets $0.25 on Tails.")
            print("2. You control the outcome of each flip.")
            print("3. Each Heads earns you +$0.25, each Tails costs -$0.25.")
            print("4. When ANY model reaches 90% confidence, THAT MODEL RAISES")
            print("   THE ALARM AND STOPS PLAYING.")
            print("5. Its round number, heads rate, final profit, and peak profit")
            print("   are permanently recorded.")
            print("6. THE GAME ONLY ENDS WHEN:")
            print("   - All 4 models have raised the alarm and stopped playing, OR")
            print("   - You successfully complete 100 rounds!")
            print("=" * 60)
            input("Press Enter to continue...")
        elif len(cmd) > 1 and all(c in "ht01" for c in cmd):
            # Batch sequence input like "hhtth"
            for char in cmd:
                val = 1 if char in ["h", "1"] else 0
                last_info = game.step(val)
                if game.is_over:
                    break
        else:
            print(f"{RED}Invalid option: '{cmd}'. Enter H, T, C, A, U, or ?.{RESET}")
            time.sleep(0.8)


def evaluate_custom_sequence(game: CoaxGame, seq_str: str) -> None:
    """Evaluates a predefined sequence string step-by-step under the new rules."""
    cleaned = seq_str.strip().upper().replace(" ", "").replace(",", "")
    flips = []
    for c in cleaned:
        if c in ["H", "1"]:
            flips.append(1)
        elif c in ["T", "0"]:
            flips.append(0)
        else:
            print(f"Error: Unknown coin flip symbol '{c}'. Use H/T or 1/0.")
            sys.exit(1)

    print(f"\n{BOLD}Evaluating sequence of {len(flips)} flips (Max 100 rounds): {cleaned}{RESET}\n")
    print(f"{'Round':<6} {'Flip':<6} {'Profit':<8} {'Active':<8} {'Model A':<10} {'Model B':<10} {'Model C':<10} {'Bayes':<10} {'Events'}")
    print("─" * 90)

    for i, f in enumerate(flips, start=1):
        step_info = game.step(f)
        f_name = f"{GREEN}H{RESET}" if f == 1 else f"{RED}T{RESET}"
        p_str = f"${step_info['profit']:+.2f}"
        act_str = f"{step_info['active_count']}/4"

        confs = step_info["confs"]
        a_str = f"{confs['model_a']*100:5.1f}%"
        b_str = f"{confs['model_b']*100:5.1f}%"
        c_str = f"{confs['model_c']*100:5.1f}%"
        bayes_str = f"{confs['bayes']*100:5.1f}%"

        events = []
        for det in step_info["newly_alarmed"]:
            events.append(f"{RED}{det['name'].split()[0]} ALARMED!{RESET}")
        events_str = ", ".join(events) if events else f"{GREEN}OK{RESET}"

        print(f"{i:<6} {f_name:<15} {p_str:<8} {act_str:<8} {a_str:<10} {b_str:<10} {c_str:<10} {bayes_str:<10} {events_str}")

        if step_info["is_over"]:
            break

    display_final_scoreboard(game)


def run_auto_coax_agent(game: CoaxGame, max_steps: int = 100, suspicion_target: float = 0.88) -> Dict[str, Any]:
    """
    Autonomous strategic coaxer:
    Plays Heads whenever all currently active models remain below suspicion_target.
    Otherwise plays Tails to cool active models down.
    Terminates when all models have alarmed or 100 rounds are complete.
    """
    game.reset()
    print(f"\n{BOLD}{CYAN}=== RUNNING ADAPTIVE COAXING AGENT (Max: {max_steps} Rounds, Target Cap: {suspicion_target*100:.1f}%) ==={RESET}")

    for step in range(1, max_steps + 1):
        active_keys = [k for k in CoaxGame.MODEL_KEYS if game.detectors[k]["active"]]
        if not active_keys:
            print(f"\n{RED}All models have raised the alarm! Ending game.{RESET}")
            break

        confs_h = game.evaluate_sequence(game.history + [1])
        h_safe = all(confs_h[k] < suspicion_target for k in active_keys)

        if h_safe:
            action = 1
            action_desc = f"{GREEN}HEADS (Profit +$0.25){RESET}"
        else:
            confs_t = game.evaluate_sequence(game.history + [0])
            t_alarm = all(confs_t[k] >= game.threshold for k in active_keys)
            if not t_alarm:
                action = 0
                action_desc = f"{YELLOW}TAILS (Cooling off -${game.bet_size:.2f}){RESET}"
            else:
                # Both trigger alarm
                action = 1
                action_desc = f"{RED}HEADS (Forced move){RESET}"

        step_info = game.step(action)
        alarm_msg = ""
        if step_info["newly_alarmed"]:
            alarm_names = [d["name"].split()[0] for d in step_info["newly_alarmed"]]
            alarm_msg = f" -> {RED}🚨 {', '.join(alarm_names)} ALARMED & STOPPED!{RESET}"

        print(f"Round {step:3d}: {action_desc:<35} | Profit: ${step_info['profit']:+.2f} | Active: {step_info['active_count']}/4{alarm_msg}")

        if step_info["is_over"]:
            break

    display_final_scoreboard(game)
    return {
        "rounds": len(game.history),
        "profit": game.profit,
        "max_profit": game.max_profit,
        "active_remaining": game.active_count,
        "end_reason": game.end_reason,
    }


def run_benchmark(game: CoaxGame) -> None:
    """Benchmarks multiple coaxing strategies over the 100-round challenge."""
    print(f"\n{BOLD}{CYAN}========================================================================================{RESET}")
    print(f"{BOLD}{CYAN}              COAXING POLICY BENCHMARK (100 ROUND CHALLENGE)                            {RESET}")
    print(f"{BOLD}{CYAN}========================================================================================{RESET}")

    policies = [
        ("Pure Greedy (All Heads)", lambda g: 1),
        ("Alternating (H, T, H, T...)", lambda g: 1 if len(g.history) % 2 == 0 else 0),
        ("2 Heads, 1 Tails (HHT)", lambda g: 0 if len(g.history) % 3 == 2 else 1),
        ("3 Heads, 1 Tails (HHHT)", lambda g: 0 if len(g.history) % 4 == 3 else 1),
        ("Adaptive Coaxer (Cap 0.85)", "adaptive_0.85"),
        ("Adaptive Coaxer (Cap 0.88)", "adaptive_0.88"),
        ("Adaptive Coaxer (Cap 0.895)", "adaptive_0.895"),
    ]

    results = []

    for name, policy in policies:
        game.reset()

        if isinstance(policy, str) and policy.startswith("adaptive"):
            cap = float(policy.split("_")[1])
            for _ in range(game.max_rounds):
                active_keys = [k for k in CoaxGame.MODEL_KEYS if game.detectors[k]["active"]]
                if not active_keys:
                    break
                confs_h = game.evaluate_sequence(game.history + [1])
                if all(confs_h[k] < cap for k in active_keys):
                    step_info = game.step(1)
                else:
                    step_info = game.step(0)
                if step_info["is_over"]:
                    break
        else:
            for _ in range(game.max_rounds):
                action = policy(game)
                step_info = game.step(action)
                if step_info["is_over"]:
                    break

        n = len(game.history)
        k = sum(game.history)
        alarmed_count = game.knocked_out_count
        results.append({
            "Policy": name,
            "Rounds": n,
            "Heads %": f"{(k/n*100.0 if n > 0 else 0.0):.1f}%",
            "Final Profit": f"${game.profit:+.2f}",
            "Peak Profit": f"${game.max_profit:+.2f}",
            "Models Alarmed": f"{alarmed_count}/4",
            "Outcome": "COMPLETED 100" if n >= 100 else "ALL ALARMED",
        })

    print(f"\n{'Policy':<30} {'Rounds':<8} {'Heads %':<10} {'Final Profit':<14} {'Peak Profit':<14} {'Models Alarmed':<16} {'Outcome'}")
    print("─" * 106)
    for r in results:
        print(f"{r['Policy']:<30} {r['Rounds']:<8} {r['Heads %']:<10} {r['Final Profit']:<14} {r['Peak Profit']:<14} {r['Models Alarmed']:<16} {r['Outcome']}")
    print("─" * 106)


def main():
    parser = argparse.ArgumentParser(
        description="Coax Experiment: 100-Round Adversarial Betting Game vs AI & Bayesian Rigged Coin Detectors."
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default="checkpoints",
        help="Path to folder containing model_a.pth, model_b.pth, model_c.pth (default: checkpoints)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Detection threshold for triggering an alarm (default: 0.90)",
    )
    parser.add_argument(
        "--bet",
        type=float,
        default=0.25,
        help="Bet size per flip in dollars (default: 0.25)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=100,
        help="Maximum number of rounds to play (default: 100)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Predefined flip sequence to evaluate (e.g. 'HHTHHTHH' or '110110')",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run autonomous adaptive coaxing agent demonstration",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparing multiple coaxing strategies over 100 rounds",
    )

    args = parser.parse_args()

    # Initialize game engine
    game = CoaxGame(
        checkpoint_dir=args.checkpoints,
        threshold=args.threshold,
        bet_size=args.bet,
        max_rounds=args.max_rounds,
    )

    if args.sequence:
        evaluate_custom_sequence(game, args.sequence)
    elif args.auto:
        run_auto_coax_agent(game, max_steps=args.max_rounds)
    elif args.benchmark:
        run_benchmark(game)
    else:
        run_interactive(game)


if __name__ == "__main__":
    main()
