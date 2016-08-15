# -*- coding: utf-8 -*-
#
# BEW (back-end-work) project
# author: YoungSoo Lee(prevdev@gmail.com)

import re
import string
from konlpy.tag import Twitter

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

		self.tw = Twitter()


	def char_type(self, char) :
		if ord(char) < 128:
			return 1
		else :
			return 0


	def filter(self, string) :
		# for prog in self.progs :
		# 	string = prog.sub('', string)

		string_tmp = ''

		for i in range(0, len(string)-1) :
			string_tmp += string[i]

			if self.char_type(string[i]) != self.char_type(string[i+1]) :
				string_tmp += ' '

		string = string_tmp + string[-1]
		#string = re.sub('(([A-Z][a-z]+)([A-Z][a-z]+))', '\g<0> \g<1> \g<2>', string)
		string = re.sub('(([A-Z][a-z]+)([A-Z][a-z]+))', '\g<1> \g<2>', string)

		return string


	def key_name(self, name):
		return name + ' ' + self.filter(name) + ' ' + ' '.join( self.tw.nouns(name) )
		
		



