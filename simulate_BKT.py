""" Dexter's code for generating BKT sequences. Joe added feature to adjust based on problem difficulty
"""

import sys
import random
import time


def forward_write(params, file_name, num_trajectories):
	fin = open(file_name, "w")
	#X = []
	Probs = []

	problem_adjustments = []
	for i in range(25):
		problem_adjustments.append(random.gauss(0,0.4))

	print problem_adjustments

	plist = range(25)

	for i in range(num_trajectories):
		random.shuffle(plist)
		Probs.append(plist[:])
		fin.write(forward(params, num_trajectories < 3, str(i), plist, problem_adjustments))
	fin.close()

	f = open('problems_' + file_name, "w")
	for ps in Probs:
		f.write(str(ps)[1:len(str(ps))-1] + '\n')
	f.close()

	g = open('adjusts_' + file_name, "w")
	g.write(str(problem_adjustments)[1:len(str(problem_adjustments))-1])
	g.close()

def forward(params, is_slow, user_id, problems, adjust):
	alpha = [0, 0]
	seq = ""
	update_probability(alpha, params, -1)
	#print params
	if is_slow:
		print "New trajectory starting!!!\n"
	for t in range(25):
		prob_correct = get_probability(alpha, params)
		if is_slow:
			print_probability(alpha, params)
		#print prob_correct
		#print prob_correct + adjust[problems[t]]
		#print
		if random.random() < prob_correct + adjust[problems[t]]:
			correct = 1
			str_correct = "CORRECT"
		else:
			correct = 0
			str_correct = "INCORRECT"
		#seq += "{0}\t{1}\t{2}\tdefault\n".format(2 - correct, user_id, t)
		seq += str(correct) + ","
		if is_slow:
			print "Got the question {0}!\n".format(str_correct)
		update_probability(alpha, params, correct)
	if is_slow:
		time.sleep(2)
	return seq[0:len(seq)-1] + "\n"


def update_probability(alpha, params, correct):
	if correct == -1:
		alpha[0] = 1 - params['pi']
		alpha[1] = params['pi']
	else:
		alpha[0] *= params['pg'] * correct + (1 - params['pg']) * (1 - correct)
		alpha[1] *= (1 - params['ps']) * correct + params['ps'] * (1 - correct)
		new_alpha = [alpha[0] * (1 - params['pt']), alpha[0] * params['pt'] + alpha[1]]
		alpha[0] = new_alpha[0]
		alpha[1] = new_alpha[1]


def print_probability(alpha, params):
	print "Probability of mastery is {0}".format(alpha[1] / (alpha[0] + alpha[1]))
	print "Probability of getting next problem correct is {0}" \
		.format((alpha[0] * params['pg'] + alpha[1] * (1 - params['ps'])) / (alpha[0] + alpha[1]))


def get_probability(alpha, params):
	return (alpha[0] * params['pg'] + alpha[1] * (1 - params['ps'])) / (alpha[0] + alpha[1])


def main():
	if len(sys.argv) < 7:
		print "python simulate_BKT.py <pi> <pt> <pg> <ps> <filename> <num_trajectories>"
		return

	pi = float(sys.argv[1])
	pt = float(sys.argv[2])
	pg = float(sys.argv[3])
	ps = float(sys.argv[4])
	file_name = sys.argv[5]
	num_trajectories = int(sys.argv[6])

	try:
		forward_write({"pi": pi, "pt": pt, "pg": pg, "ps": ps}, file_name, num_trajectories)
	except KeyboardInterrupt:
		return


if __name__ == "__main__":
	main()
