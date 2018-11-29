#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy2 Experiment Builder (v1.90.3),
    on November 17, 2018, at 18:59
If you publish work using this script please cite the PsychoPy publications:
    Peirce, JW (2007) PsychoPy - Psychophysics software in Python.
        Journal of Neuroscience Methods, 162(1-2), 8-13.
    Peirce, JW (2009) Generating stimuli for neuroscience using PsychoPy.
        Frontiers in Neuroinformatics, 2:10. doi: 10.3389/neuro.11.010.2008
"""

from __future__ import absolute_import, division
from psychopy import locale_setup, sound, gui, visual, core, data, event, logging, clock
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)
import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle
import os  # handy system and path functions
import sys  # to get file system encoding

# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__)).decode(sys.getfilesystemencoding())
os.chdir(_thisDir)

# Store info about the experiment session
expName = u'Sensor Calibration'  # from the Builder filename that created this script
expInfo = {u'gender': u'', u'age': u'', u'name': u''}
dlg = gui.DlgFromDict(dictionary=expInfo, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + 'data/%s_%s_%s' %(expInfo['name'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath=u'C:\\Users\\ermay\\OneDrive\\Documents\\Bias\\ImplicitAssociationTest-1.2.1\\ImplicitAssociationTest-1.2.1\\IAT.psyexp',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.WARNING)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp

# Start Code - component code to be run before the window creation

# Setup the Window
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0,
    allowGUI=False, allowStencil=False,
    monitor=u'testMonitor', color=[-1.000,-1.000,-1.000], colorSpace='rgb',
    blendMode='avg', useFBO=True)
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Initialize components for Routine "instruction"
instructionClock = core.Clock()
# Dependencies
import itertools  # for flattening lists of lists into lists
import random
import math  # for math.ceil() rounding up
import serial
import time

ser = serial.Serial(port="COM5", baudrate=19200)
testNumber = 0

# Import stimuli exemplars
exemplars_filename = 'stimuli.xlsx'
exemplars = data.importConditions(exemplars_filename)# Import stimuli exemplars

# Determine rows of examplars (i.e., max number of rows)
"""
This method creates a fully counterbalanced presentation of exemplars when there are 5
of them, but it will not present each one an equal number of times it the n diverges from 5.
"""
n_exemplars = len(exemplars)
list_multiplier = int(math.ceil(10/n_exemplars))  # math.ceil() rounds up. 10 is the derived from way block lengths are calculated. Admittedly, this comment doensn't adequately document why it's ten. Honestly, I have to work it out of my fingers every time and can't explain it.

# Trial generation function
def generate_trials(trial_type_column, multiplier):
    """Generate a shuffled list of stimuli exemplars from a column in an excel stimuli file""" 
    a = dict()  # declare a dict to be populated
    for i in range(len(exemplars)):
        a[i] = [exemplars[i][trial_type_column]] * multiplier  # populate the dict from vertical reads of the conditions
    a = a.values()  # extract only values (and not keys) from the list of dicts
    a = list(itertools.chain(*a))  # flatten the list of dicts into a list
    random.shuffle(a)  # shuffle this list, so that it can be drawn from by the trials
    return a

# declare trial rows (not sure if necessary, but can't be bothered to removed and test)
trial_rows = ""


block_order = 1

instructionsBox = visual.TextStim(win=win, name='instructionsBox',
    text='default text',
    font='Arial',
    pos=[0, 0], height=0.05, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-1.0);
leftCategoryLabel_2 = visual.TextStim(win=win, name='leftCategoryLabel_2',
    text='default text',
    font='Arial',
    pos=[-.6, .85], height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-3.0);
rightCategoryLabel_2 = visual.TextStim(win=win, name='rightCategoryLabel_2',
    text='default text',
    font='Arial',
    pos=[.6, .85], height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-4.0);
leftAttributeLabel_2 = visual.TextStim(win=win, name='leftAttributeLabel_2',
    text='default text',
    font='Arial',
    pos=[-.6, 0.55], height=0.1, wrapWidth=None, ori=0, 
    color=[-1, 1, -1], colorSpace='rgb', opacity=1,
    depth=-5.0);
rightAttributeLabel_2 = visual.TextStim(win=win, name='rightAttributeLabel_2',
    text='default text',
    font='Arial',
    pos=[.6, 0.55], height=0.1, wrapWidth=None, ori=0, 
    color=[-1, 1, -1], colorSpace='rgb', opacity=1,
    depth=-6.0);
orLeft_2 = visual.TextStim(win=win, name='orLeft_2',
    text='default text',
    font='Arial',
    pos=[-.6, .7], height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-7.0);
orRight_2 = visual.TextStim(win=win, name='orRight_2',
    text='default text',
    font='Arial',
    pos=[.6, .7], height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=-8.0);

# Initialize components for Routine "trial"
trialClock = core.Clock()
#declare accuracy feedback message variable
msg=""
name = expInfo['name']
biasName = "calibrationData%s.txt" %name
biasFile = open(biasName, "w")
#print("hello")
time.sleep(3)

stimulusImageBox = visual.ImageStim(
    win=win, name='stimulusImageBox',
    image='sin', mask=None,
    ori=0, pos=[0, 0], size=.6,
    color=[1,1,1], colorSpace='rgb', opacity=1,
    flipHoriz=False, flipVert=False,
    texRes=128, interpolate=True, depth=-1.0)
stimulusTextBox = visual.TextStim(win=win, name='stimulusTextBox',
    text='default text',
    font=u'Arial',
    pos=[0, 0], height=0.1, wrapWidth=None, ori=0, 
    color=1.0, colorSpace='rgb', opacity=1,
    depth=-2.0);

# Initialize components for Routine "end"
endClock = core.Clock()
endBox = visual.TextStim(win=win, name='endBox',
    text='End of the task',
    font='Arial',
    pos=[0, 0], height=0.1, wrapWidth=None, ori=0, 
    color='white', colorSpace='rgb', opacity=1,
    depth=0.0);

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# set up handler to look after randomisation of conditions etc
blocks = data.TrialHandler(nReps=1, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('instructions.xlsx'),
    seed=None, name='blocks')
thisExp.addLoop(blocks)  # add the loop to the experiment
thisBlock = blocks.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
if thisBlock != None:
    for paramName in thisBlock:
        exec('{} = thisBlock[paramName]'.format(paramName))

for thisBlock in blocks:
    currentLoop = blocks
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            exec('{} = thisBlock[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "instruction"-------
    t = 0
    instructionClock.reset()  # clock
    frameN = -1
    continueRoutine = True
    # update component parameters for each repeat
    # set the block length and the rows to pull from based on the current block 
    # this layout follows Nosek et al. 2007, "The Implicit Association Test at age 7: A methodological and conceptual review"
    if blocks.thisN == 0:
        trial_rows = "0:2" 
        n_block_repeats = 10   #2*10 = 20 trials
        modified_list_multiplier = list_multiplier
    elif blocks.thisN == 1:
        trial_rows = "2:4" 
        n_block_repeats = 10   #2*10 = 20 trials
        modified_list_multiplier = list_multiplier
    elif blocks.thisN == 2:
        trial_rows = "0:4" 
        n_block_repeats = 5   #4*5 = 20 trials
        modified_list_multiplier = list_multiplier
    elif blocks.thisN == 3:
        trial_rows = "0:4" 
        n_block_repeats = 10   #4*10 = 40 trials
        modified_list_multiplier = list_multiplier
    elif blocks.thisN == 4:
        trial_rows = "0:2" 
        n_block_repeats = 20   #2*20 = 40 trials
        modified_list_multiplier = list_multiplier * 2  # because this block has a different trials:categories ratio
    elif blocks.thisN == 5:
        trial_rows = "0:4" 
        n_block_repeats = 5   #4*5 = 20 trials
        modified_list_multiplier = list_multiplier
    elif blocks.thisN == 6:
        trial_rows = "0:4" 
        n_block_repeats = 10   #4*10 = 40 trials
        modified_list_multiplier = list_multiplier
    
    # Generate list of stimuli for the block
    text_trial_type_1_trials = generate_trials('text_trial_type_1_exemplars', modified_list_multiplier)  # function and variable determined at begin exp.
    text_trial_type_2_trials = generate_trials('text_trial_type_2_exemplars', modified_list_multiplier)
    text_trial_type_3_trials = generate_trials('text_trial_type_3_exemplars', modified_list_multiplier)
    text_trial_type_4_trials = generate_trials('text_trial_type_4_exemplars', modified_list_multiplier)
    img_trial_type_1_trials = generate_trials('img_trial_type_1_exemplars', modified_list_multiplier)
    img_trial_type_2_trials = generate_trials('img_trial_type_2_exemplars', modified_list_multiplier)
    img_trial_type_3_trials = generate_trials('img_trial_type_3_exemplars', modified_list_multiplier)
    img_trial_type_4_trials = generate_trials('img_trial_type_4_exemplars', modified_list_multiplier)
    
    # set category and attribute labels based on the block order and current block
    if block_order == 1 and blocks.thisN <= 3:
        leftCategory = categoryA
        rightCategory = categoryB
    elif block_order == 1 and blocks.thisN > 3:
        leftCategory = categoryB
        rightCategory = categoryA
    elif block_order == 2 and blocks.thisN <= 3:
        leftCategory = categoryB
        rightCategory = categoryA
    elif block_order == 2 and blocks.thisN > 3:
        leftCategory = categoryA
        rightCategory = categoryB
    instructionsBox.setText(instructions)
    instructionsKey = event.BuilderKeyResponse()
    leftCategoryLabel_2.setText(leftCategory)
    rightCategoryLabel_2.setText(rightCategory)
    leftAttributeLabel_2.setText(leftAttribute)
    rightAttributeLabel_2.setText(rightAttribute)
    orLeft_2.setText(orStimulus)
    orRight_2.setText(orStimulus)
    # keep track of which components have finished
    instructionComponents = [instructionsBox, instructionsKey, leftCategoryLabel_2, rightCategoryLabel_2, leftAttributeLabel_2, rightAttributeLabel_2, orLeft_2, orRight_2]
    for thisComponent in instructionComponents:
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    
    # -------Start Routine "instruction"-------
    while continueRoutine:
        # get current time
        t = instructionClock.getTime()
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        
        # *instructionsBox* updates
        if t >= 0.75 and instructionsBox.status == NOT_STARTED:
            # keep track of start time/frame for later
            instructionsBox.tStart = t
            instructionsBox.frameNStart = frameN  # exact frame index
            instructionsBox.setAutoDraw(True)
        
        # *instructionsKey* updates
        if t >= 2.75 and instructionsKey.status == NOT_STARTED:
            # keep track of start time/frame for later
            instructionsKey.tStart = t
            instructionsKey.frameNStart = frameN  # exact frame index
            instructionsKey.status = STARTED
            # keyboard checking is just starting
            event.clearEvents(eventType='keyboard')
        if instructionsKey.status == STARTED:
            theseKeys = event.getKeys(keyList=['e', 'i'])
            
            # check for quit:
            if "escape" in theseKeys:
                endExpNow = True
            if len(theseKeys) > 0:  # at least one key was pressed
                # a response ends the routine
                continueRoutine = False
        
        # *leftCategoryLabel_2* updates
        if t >= 0.75 and leftCategoryLabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            leftCategoryLabel_2.tStart = t
            leftCategoryLabel_2.frameNStart = frameN  # exact frame index
            leftCategoryLabel_2.setAutoDraw(True)
        
        # *rightCategoryLabel_2* updates
        if t >= 0.75 and rightCategoryLabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            rightCategoryLabel_2.tStart = t
            rightCategoryLabel_2.frameNStart = frameN  # exact frame index
            rightCategoryLabel_2.setAutoDraw(True)
        
        # *leftAttributeLabel_2* updates
        if t >= 0.75 and leftAttributeLabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            leftAttributeLabel_2.tStart = t
            leftAttributeLabel_2.frameNStart = frameN  # exact frame index
            leftAttributeLabel_2.setAutoDraw(True)
        
        # *rightAttributeLabel_2* updates
        if t >= 0.75 and rightAttributeLabel_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            rightAttributeLabel_2.tStart = t
            rightAttributeLabel_2.frameNStart = frameN  # exact frame index
            rightAttributeLabel_2.setAutoDraw(True)
        
        # *orLeft_2* updates
        if t >= 0.75 and orLeft_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            orLeft_2.tStart = t
            orLeft_2.frameNStart = frameN  # exact frame index
            orLeft_2.setAutoDraw(True)
        
        # *orRight_2* updates
        if t >= 0.75 and orRight_2.status == NOT_STARTED:
            # keep track of start time/frame for later
            orRight_2.tStart = t
            orRight_2.frameNStart = frameN  # exact frame index
            orRight_2.setAutoDraw(True)
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructionComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # check for quit (the Esc key)
        if endExpNow or event.getKeys(keyList=["escape"]):
            core.quit()
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "instruction"-------
    for thisComponent in instructionComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=n_block_repeats, method='fullRandom', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions('block_layout.xlsx', selection=trial_rows),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    for thisTrial in trials:
        currentLoop = trials
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "trial"-------
        t = 0
        trialClock.reset()  # clock
        frameN = -1
        continueRoutine = True
        routineTimer.add(10.300000)
        # update component parameters for each repeat
        
        #import ctypes  # An included library with Python install.   
        #ctypes.windll.user32.MessageBoxW(0, "Your text", "Your title", 1)
        
        
        # choose a random exemplar from the appropriate trial type list
        if trial_type == 1:
            text_stimulus = text_trial_type_1_trials.pop()
            image_stimulus = img_trial_type_1_trials.pop()
        elif trial_type == 2:
            text_stimulus = text_trial_type_2_trials.pop()
            image_stimulus = img_trial_type_2_trials.pop()
        elif trial_type == 3:
            text_stimulus = text_trial_type_3_trials.pop()
            image_stimulus = img_trial_type_3_trials.pop()
        elif trial_type == 4:
            text_stimulus = text_trial_type_4_trials.pop()
            image_stimulus = img_trial_type_4_trials.pop()
        
        
        biasFile.write(image_stimulus)
        biasFile.write("\n")
        ser.write("f")
        
        # set stimulus colors based on trial type
        if trial_type == 1 or trial_type == 2:
             stimulusColor = [1, 1, 1]
        elif trial_type >2:
             stimulusColor = [-1, 1, -1]
        stimulusImageBox.setImage(image_stimulus)
        stimulusTextBox.setColor(stimulusColor, colorSpace='rgb')
        stimulusTextBox.setText(text_stimulus)
        # keep track of which components have finished
        trialComponents = [stimulusImageBox, stimulusTextBox]
        for thisComponent in trialComponents:
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        
        # -------Start Routine "trial"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = trialClock.getTime()
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            
            if ser.in_waiting > 0:
                biasFile.write(ser.readline())
            
            
            # *stimulusImageBox* updates
            if t >= 0.3 and stimulusImageBox.status == NOT_STARTED:
                # keep track of start time/frame for later
                stimulusImageBox.tStart = t
                stimulusImageBox.frameNStart = frameN  # exact frame index
                stimulusImageBox.setAutoDraw(True)
            frameRemains = 0.3 + 10- win.monitorFramePeriod * 0.75  # most of one frame period left
            if stimulusImageBox.status == STARTED and t >= frameRemains:
                stimulusImageBox.setAutoDraw(False)
            
            # *stimulusTextBox* updates
            if t >= 0.3 and stimulusTextBox.status == NOT_STARTED:
                # keep track of start time/frame for later
                stimulusTextBox.tStart = t
                stimulusTextBox.frameNStart = frameN  # exact frame index
                stimulusTextBox.setAutoDraw(True)
            frameRemains = 0.3 + 10- win.monitorFramePeriod * 0.75  # most of one frame period left
            if stimulusTextBox.status == STARTED and t >= frameRemains:
                stimulusTextBox.setAutoDraw(False)
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # check for quit (the Esc key)
            if endExpNow or event.getKeys(keyList=["escape"]):
                core.quit()
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial"-------
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        ser.write("f")
        while ser.in_waiting > 0:
            biasFile.write(ser.readline())
            
        biasFile.write("endOfImage\n")
        
        thisExp.nextEntry()
        
    # completed n_block_repeats repeats of 'trials'
    
# completed 1 repeats of 'blocks'


# ------Prepare to start Routine "end"-------
t = 0
endClock.reset()  # clock
frameN = -1
continueRoutine = True
routineTimer.add(3.750000)
# update component parameters for each repeat
# keep track of which components have finished
endComponents = [endBox]
for thisComponent in endComponents:
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED

# -------Start Routine "end"-------
while continueRoutine and routineTimer.getTime() > 0:
    # get current time
    t = endClock.getTime()
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *endBox* updates
    if t >= 0.75 and endBox.status == NOT_STARTED:
        # keep track of start time/frame for later
        endBox.tStart = t
        endBox.frameNStart = frameN  # exact frame index
        endBox.setAutoDraw(True)
    frameRemains = 0.75 + 3- win.monitorFramePeriod * 0.75  # most of one frame period left
    if endBox.status == STARTED and t >= frameRemains:
        endBox.setAutoDraw(False)
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in endComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # check for quit (the Esc key)
    if endExpNow or event.getKeys(keyList=["escape"]):
        core.quit()
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "end"-------
for thisComponent in endComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
ser.close()

testNumber = testNumber + 1
biasFile.close()
# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
