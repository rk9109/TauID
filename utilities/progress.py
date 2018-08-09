import sys, string

def update_progress(progress):
	"""
	Return: Progress Bar
	Input: progress | Fraction completed from 0->1
	"""
	barLength = 20
	status = ''

	if (progress >= 1):
		progress = 1
		status = "Done...\r\n"

	block = int(round(barLength*progress))
	text = '\rPercent: [{0}] {1}% {2}'.format('#'*block + '-'*(barLength - block), progress*100, status)
	sys.stdout.write(text)
	sys.stdout.flush()

def update_progress_inline(message, progress):
	"""
	Return: Progress Statement
	Input: progress | Fraction completed from 0->1
	"""
	status = ''

	if (progress >= 1):
		progress = 100
		status = "Done!\r\n"
	
	text = '\r{0}...{1}% {2}'.format(message, progress*100, status)
	sys.stdout.write(text)
	sys.stdout.flush()

