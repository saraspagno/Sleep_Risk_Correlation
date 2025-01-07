import time
from datetime import datetime

# Edit here to calculate for different groups
GROUPS = ["first_50", "second_50"]

# For the risk scores, either use the objective reward of each image,
# or use a learned perceived score that the user learned from his past choices.
USE_PERCEIVED_REWARD = True
SKIP_IF_RANK_NOT_1 = True

'''
Selects all the sleep answers from the ANSWERS table including the score "overall".
'''
SLEEP_QUERY = """
SELECT answer, answer_time 
FROM answers 
WHERE questionnaire_name ='יומן שינה' 
AND answer LIKE '%overall=%'
"""

'''
Selects all information from TRIALS table, joined on the STIMULI table based on equal image number.
'''
RISK_QUERY = """
SELECT 
    stim1_stimuli.reward AS r_0, stim2_stimuli.reward AS r_1,
    trials.choice, trials.choice_time, 
    trials.stim1 as im_0, trials.stim2 as im_1,
    trials.outcome, trials.feedback, stim1_stimuli.rank, stim2_stimuli.rank,
    trials.trial
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


MOOD_QUERY = """
SELECT answer, answer_time 
FROM answers 
WHERE answer LIKE '%Anxious=%' OR answer LIKE'%Valence=%'
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
