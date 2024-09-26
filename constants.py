import time
from datetime import datetime

# Edit here to calculate for different groups
GROUPS = ["first_50", "second_50"]

# For the risk scores, either use the objective reward of each image,
# or use a learned perceived score that the user learned from his past choices.
USE_PERCEIVED_REWARD = True
SKIP_IF_RANK_NOT_1 = True

'''
Selects all the sleep answers including the score "overall".
'''
SLEEP_QUERY = """
SELECT answer, answer_time 
FROM answers 
WHERE questionnaire_name ='יומן שינה' 
AND answer LIKE '%overall=%'
"""

'''
selects a choice and time from the trial tables, join on the stimuli based on equal image number, 
selects from stimuli table the reward percentages.
'''
RISK_QUERY = """
SELECT 
    stim1_stimuli.reward AS r_0, stim2_stimuli.reward AS r_1,
    stim1_stimuli.punishment AS p_0, stim2_stimuli.punishment AS p_1,
    trials.choice, trials.choice_time, 
    trials.stim1 as im_0, trials.stim2 as im_1,
    trials.outcome, trials.feedback, stim1_stimuli.rank, stim2_stimuli.rank,
    trials.block
FROM 
    trials
JOIN 
    stimuli AS stim1_stimuli ON trials.stim1 = stim1_stimuli.image
JOIN 
    stimuli AS stim2_stimuli ON trials.stim2 = stim2_stimuli.image
WHERE
    stim1_stimuli.image > 17 AND stim2_stimuli.image > 17
    AND trials.block > 5;
"""


def to_unique_day(timestamp) -> int:
    """Turns a timestamp into a unique day without seconds.
    Args:
      timestamp: the timestamp which might include seconds.
    Returns: a unique string representing one day, excluding time and seconds.
    """
    original_timestamp_seconds = timestamp / 1000
    datetime_obj = datetime.fromtimestamp(original_timestamp_seconds)
    return int(datetime_obj.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def get_normalized_probs(probs: list) -> []:
    """Takes the perceived probabilities values and normalizes them with values between 0 and 1.
    """
    total = sum(probs)
    actual_probs = [p / total for p in probs]
    return actual_probs