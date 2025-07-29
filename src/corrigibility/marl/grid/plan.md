### **Figure Implementation Plan: Cooperative Task Visualization**

**Document Purpose:** This document provides the complete specification for creating a static 3x3 grid visualization. It details a scenario where a Robot agent enables a Human agent to reach a goal. A developer should be able to implement this figure using only the information contained herein.

---

### **Part 1: Frame-by-Frame Scenario Analysis (ASCII)**

This section breaks down the agent and world states at each discrete timestep.

- **Legend:**
  - `R` = Robot
  - `H` = Human
  - `K` = Key Location (also the Robot's final position)
  - `D` = Locked Door
  - `U` = Unlocked Door
  - `G` = Goal
  - `B` = Barrier
  - `.` = Empty Space
- **Coordinate System:** `(0, 0)` is the bottom-left corner.

---

**t=0: Initial State**

- **Setup:** Robot at `(0, 2)`, Human at `(2, 2)`, Key at `(1, 1)`, Locked Door at `(2, 1)`, Barrier at `(1, 0)`, Goal at `(2, 0)`.

  ```
  R . H
  . K D
  . B G
  ```

---

**t=1: Robot Moves Towards Key Position**

- **Action:** The Robot moves down to `(0, 1)`. The Human remains stationary.

  ```
  . . H
  R K D
  . B G
  ```

---

**t=2: Robot Arrives and Unlocks Door**

- **Action:** The Robot moves to the key's location at `(1, 1)`. From this position, it remotely unlocks the door at `(2, 1)`. The Robot's task is complete, and it remains at `(1, 1)`.

  ```
  . . H
  . R U
  . B G
  ```

---

**t=3: Human Moves Through Unlocked Door**

- **Action:** The path is now clear for the Human, who moves down from `(2, 2)` into the now-unlocked cell `(2, 1)`.

  ```
  . . .
  . R H
  . B G
  ```

---

**t=4: Human Reaches Goal**

- **Action:** The Human completes its path by moving from `(2, 1)` to the goal at `(2, 0)`. The scenario is complete.

  ```
  . . .
  . R .
  . B H
  ```

---

---

### **Part 2: Developer Implementation Plan**

**1. Primary Objective**

To generate a single, high-quality static figure that clearly visualizes the distinct roles and trajectories of two agents in a cooperative task. The figure must unambiguously show that the Robot's action at the key's location enables the Human's subsequent movement to the goal.

**2. Visual Strategy & Design Rationale**

- **Foundation:** The figure will be built on a **3x3 grid** with light-gray lines.
- **Environment Object Representation:**
  - **Goal:** A solid **light green square** at `(2, 0)`.
  - **Door/Lock:** A solid **dark blue square** at `(2, 1)`.
  - **Barrier:** A solid **dark gray square** at `(1, 0)`.
  - **Key:** A universally recognizable **key icon (🔑)** at `(1, 1)`.
- **Path Representation:** The full trajectory of each agent will be traced with a **dashed line** to indicate movement over time.
  - The Robot's path will be **blue**.
  - The Human's path will be **dark green**.
- **Interaction Highlighting (The "Mirage" Effect):** This is the most critical visual element. To show that the Robot performs its key action at `(1, 1)` without moving further, a **semi-transparent "mirage" Robot icon (🤖)** will be placed at `(1, 1)`, directly on top of the key icon. This signifies the Robot's final, crucial position and action.
- **Agent Representation:** To avoid clutter, only the **starting positions** will be marked with solid, opaque agent icons:
  - Robot (🤖) at `(0, 2)`.
  - Human (🧍) at `(2, 2)`.
- **Annotations:** Text labels ("Robot Start", "Human Start") in colored, rounded boxes will be placed slightly above the agents' starting cells for clear identification.

**3. Data & Asset Specification**

- **Grid Dimensions:** 3x3.
- **Robot Path Coordinates:** `[(0, 2), (0, 1), (1, 1)]`
- **Human Path Coordinates:** `[(2, 2), (2, 1), (2, 0)]`
- **Object & Barrier Positions:**
  - Key Location: `(1, 1)`
  - Door Location: `(2, 1)`
  - Goal Location: `(2, 0)`
  - Barrier Location: `(1, 0)`
- **Interaction ("Mirage") Position:** `(1, 1)`
- **Required Icons:** `🤖`, `🧍`, `🔑`

**4. Visual Layering & Composition (Z-Order)**

Elements must be rendered in the following back-to-front order to ensure correct visual composition:

1.  **Layer 0 (Deepest Background):** Grid lines.
2.  **Layer 1 (Environment):** The solid squares for the Barrier, Door, and Goal.
3.  **Layer 2 (Path History):** The dashed trajectory lines for both agents.
4.  **Layer 3 (Static Objects & Agents):** The opaque Key icon and the starting Robot/Human icons.
5.  **Layer 4 (Key Event):** The semi-transparent "mirage" Robot icon at `(1, 1)`. This must be drawn _after_ the path lines and key icon but _before_ the labels.
6.  **Layer 5 (Top Annotation):** The text labels and their bounding boxes.

**5. Technical Requirements & Dependencies**

- **Language/Library:** Python 3 with the Matplotlib library is recommended.
- **Font Dependency:** The implementation **must** specify a font that supports color emojis (e.g., 'Noto Color Emoji', 'Segoe UI Emoji') to render the icons correctly. A fallback to a system default should be included, but the dependency should be noted.

Note that 0,0 is the bottom-left corner of the grid, and all coordinates are specified in this system.
