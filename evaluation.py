from sys import stderr, stdout
from re import sub
from pprint import pformat

def print_dict(obj, file=stdout):
    print(sub('^{|\'|\n|}$', '', pformat(obj)), file=file)


def score(te, result):
    # data {{{
    fee = 1
    stop_loss = 20
    return_roi = []

    log = { # {{{
        'actions': [],
        '#2action_day': 0, # temporary variable
        'consecutive_loss': 0, # temporary variable
        '#3consecutive_loss_max': 0,
        'consecutive_loss_days': 0, # temporary variable
        '#4consecutive_loss_days_max': 0,
        '#5loss_days': 0,
        'roi': 0,
        'roi_max': 0,
        'roi_min': 0,
    } # }}}

    for z, row in zip(te, result.iterrows()): # simulation {{{
        roi = 0
        r = row[1]
        time = "00:00"
        if(z == 1):
            roi = r['Call_result']
            time = r['Call_trading_time']
        elif(z == -1):
            roi = r['Put_result']
            time = r['Put_trading_time']
        profit = 0
        if(0 != roi):
            profit =  int(roi - fee)
            log['#2action_day'] += 1
        
            if profit < 0:
                log['consecutive_loss'] += profit
                log['consecutive_loss_days'] += 1
                log['#5loss_days'] += 1
            else:
                if log['consecutive_loss'] < log['#3consecutive_loss_max']:
                    log['#3consecutive_loss_max'] = log['consecutive_loss']
                if log['consecutive_loss_days'] > log['#4consecutive_loss_days_max']:
                    log['#4consecutive_loss_days_max'] = log['consecutive_loss_days']
                log['consecutive_loss'] = 0
                log['consecutive_loss_days'] = 0
        log['roi'] += profit
        if log['roi'] > log['roi_max']: log['roi_max'] = log['roi']
        if log['roi'] < log['roi_min']: log['roi_min'] = log['roi']
        # if None != trade_time:
        log['actions'].append((r['Date'], z, profit, log['roi'], log['roi'] * 200 + 100000, time))
        # }}}
    # output {{{
    for action in log['actions']:
        print(','.join(map(str, action)))
        return_roi.append(action[3])

    del log['actions']
    del log['consecutive_loss'], log['consecutive_loss_days']
    print_dict(log)
    print_dict(log, file=stderr)

    return return_roi
    # }}}