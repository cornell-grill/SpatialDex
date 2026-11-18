PICK_BALL = "Pick up the red ball and place it in the gray box-shaped container"
SWEEP_BROOM = "Use broom to sweep the {obj} into the dustpan"
SWEEP_DUSTPAN = "Hold the dustpan to receive the {obj}"
HANDOVER_FIRST = "Pick up the {obj} and hand it off to the other hand"
HANDOVER_SECOND = "Receive the {obj} from the other hand and place it in the gray box-shaped container"
OBJECT_NAMES = {"rubberduck_convex": "yellow duck", "pringles_can": "red pringles can"}


def episode_instruction(source, label):
    task = source.attrs["task"]
    if task == "pick_ball":
        active = (label in ("L", "B"), label in ("R", "B"))
        left = PICK_BALL if active[0] else "None"
        right = PICK_BALL if active[1] else "None"
        return f"Left hand: {left}. Right hand: {right}", active
    obj = OBJECT_NAMES[source.attrs["selected_object_name"]]
    if task == "sweep":
        first, second = SWEEP_BROOM.format(obj=obj), SWEEP_DUSTPAN.format(obj=obj)
    else:
        first, second = HANDOVER_FIRST.format(obj=obj), HANDOVER_SECOND.format(obj=obj)
    left, right = (first, second) if label == "L" else (second, first)
    return f"Left hand: {left}. Right hand: {right}", (True, True)
