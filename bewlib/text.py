# -*- coding: utf-8 -*-
# Author: YoungSoo Lee(prevdev@gmail.com)

import re
import string


class Filter() :

	def __init__(self) :
		self.ignore_words = (
			'백화점', '할인', '쿠폰', '카드', '정품', '해외', '구매',
			'배송', '이베이', '아마존', 'Mall', '몰', '플라자'
		)

		self.progs = []

		for word in self.ignore_words :
			self.progs.append( re.compile('\([^\)]*'+word+'.*?\)') )
			self.progs.append( re.compile('\[[^\]]*'+word+'.*?\]') )


	def char_type(self, char) :
		# for i in list(' -_\\/') :
		# 	if char == i :
		# 		return 1

		if char == ' ' :
			return 1

		for i in list(string.ascii_lowercase + string.ascii_uppercase) :
			if char == i :
				return 2

		if char.isdigit() :
			return 2

		return 0


	def filter(self, string) :
		for prog in self.progs :
			string = prog.sub('', string)

		string = re.sub('[\[\]\(\)]', ' ', string)
		string_tmp = ''

		for i in range(0, len(string)-1) :
			string_tmp += string[i]

			if self.char_type(string[i]) != self.char_type(string[i+1]) :
				string_tmp += ' '

		string = string_tmp + string[-1]

		string = re.sub('( ){2,}', ' ', string)
		string = string.strip()

		return string


