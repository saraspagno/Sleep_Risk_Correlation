from datetime import datetime

# Edit here to calculate for different groups
GROUPS = ["first_50", "second_50"]

# For the risk scores, either use the objective reward of each image,
# or use a learned perceived score that the user learned from his past choices.
USE_PERCEIVED_REWARD = True

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
    trials.choice, trials.choice_time,
    trials.stim1 as im_0, trials.stim2 as im_1,
    trials.outcome
FROM 
    trials
JOIN 
    stimuli AS stim1_stimuli ON trials.stim1 = stim1_stimuli.image
JOIN 
    stimuli AS stim2_stimuli ON trials.stim2 = stim2_stimuli.image
WHERE
    stim1_stimuli.image > 17 AND stim2_stimuli.image > 17
    AND stim1_stimuli.rank == 1 AND stim2_stimuli.rank == 1
    AND trials.feedback = true AND trials.trial > 5;
"""


def to_unique_day(timestamp):
    """Turns a timestamp into a unique day without seconds.
    Args:
      timestamp: the timestamp which might include seconds.
    Returns: a unique string representing one day, excluding time and seconds.
    """
    original_timestamp_seconds = timestamp / 1000
    datetime_obj = datetime.fromtimestamp(original_timestamp_seconds)
    return datetime_obj.strftime("%d %B %Y")