# finetuned on speculation F1

finetuned_baseline = {'?': ('backward', 0.6687898089171974),
 'eventueel': ('forward', 0.5925925925925926),
 'eventuele': ('forward', 0.36363636363636365),
 'kan': ('forward', 0.18666666666666668),
 'mogelijk': ('forward', 0.41666666666666663),
 'mogelijke': ('forward', 0.2857142857142857),
 'onduidelijk': ('forward_punct', 1.0),
 'onwaarschijnlijk': ('backward_punct', 0.6666666666666666),
 'vermoedelijk': ('forward', 0.6666666666666667),
 'vermoeden': ('forward', 0.7741935483870968),
 'waarschijnlijk': ('forward', 0.5),
 'waarschijnlijke': ('backward', 1.0)}

finetuned_hybrid = {'?': (('dependency', True, 1, False, True), 0.7876447876447876),
 'eventueel': (('forward', True), 0.5925925925925926),
 'eventuele': (('dependency', False, 1, False, True), 0.4),
 'kan': (('forward', False), 0.1917808219178082),
 'mogelijk': (('forward', False), 0.425531914893617),
 'mogelijke': (('dependency', False, 1, False, True), 0.4),
 'onduidelijk': (('forward_punct', False), 1.0),
 'onwaarschijnlijk': (('backward', False), 0.6666666666666666),
 'vermoedelijk': (('forward', True), 0.6666666666666667),
 'vermoeden': (('forward', False), 0.793103448275862),
 'waarschijnlijk': (('forward', True), 0.5),
 'waarschijnlijke': (('backward', False), 1.0)}
