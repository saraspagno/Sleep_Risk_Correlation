# edit here to calculate for different groups
GROUP = "first_50"

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
    trials.choice, trials.choice_time
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
