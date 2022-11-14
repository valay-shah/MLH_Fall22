#!/usr/bin/env python

"""File containing function to reading and performing processing on the report."""

def tokenized_session(session):
    """
    Remove unwanted chars and tokenize the session.

    Parameter:
    ----------
    session: list
        A selected / sliced session from a report. i.e. it can be either FINDINGS or IMPRESSION.

    Returns:
    -------
    tk_session: list
        The tokenzied session
    """
    tk_session = ''

    #Remove the unwanted char
    for line in session:
        line = re.sub(r'[\n,.]', '', line)
        line = re.sub(r'^ ', '', line)
        tk_session += line

    #Tokenize
    tk_session = tk_session.split(' ')

    return tk_session

def findings_impression(data_path):
    """
    Slice FINDINDS and IMPRESSION from a radiology report

    Parameter:
    ----------
    data_path: file path
        A file path to ONE radiology report in .txt format.

    Returns:
    -------
    tk_findings, tk_impressions: list, list
        Tokenzied FINDINGS and IMPRESSION
    """
    #Read a radiology report
    with open(data_path) as f:
        lines = f.readlines()

    #Find where FINDINGS and IMPRESSIONS start in a report
    f, imp = None, None
    for i in range(len(lines)):
        line = lines[i]
        if 'FINDINGS' in line:
            f = i
        if 'IMPRESSION' in line:
            imp = i

    #Slice FINDINGS and IMPRESSIONS from the report
    findings = lines[f+2, imp-1]
    impression = lines[imp+2:]

    return tokenized_session(findings), tokenized_session(impression)

def shuffle():
    pass

    

