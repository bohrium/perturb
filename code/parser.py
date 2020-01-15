''' author: samtenka
    change: 2020-01-15
    create: 2020-01-15
    descrp: evaluate expressions in a concise mathematical mini-language
'''


from utils import pre, prod
import numpy as np

def factorial(n):
    n = int(n)
    return prod(range(1, n+1))

def choose(T, t):
    T, t = int(T), int(t)
    return int(prod(range(T-t+1, T+1)) / float(prod(range(1, t+1))))

class MathParser(object):
    '''
        Parse expressions of this grammar:
            EXPR = (empty | TERM) ((+ | -) TERM)* 
            etc
    '''
    def __init__(self, s='', vals_by_name = {}):
        self.refresh(s, vals_by_name)

    def refresh(self, s, vals_by_name):
        self.s = s.strip()
        self.i = 0
        self.vals_by_name = vals_by_name

    def eval(self, s, vals_by_name={}):
        self.refresh(s, vals_by_name)
        return self.eval_expr()

    def at_end(self):
        return self.i == len(self.s)
    def peek(self):
        pre(not self.at_end(), 'attempted to peek beyond end of string!')
        return self.s[self.i] 
    def match(self, expected_c):
        c = self.peek()
        pre(c==expected_c, 'expected {} but got {}'.format(c, expected_c))
        self.i += 1
    def skip_white(self):
        while (not self.at_end()) and self.peek()==' ':
            self.match(self.peek())

    def eval_expr(self): 
        if self.peek() in '+-':
            val = 0
        else:
            val = self.eval_term()

        self.skip_white()
        while not self.at_end():
            self.skip_white()
            if self.peek() == '+':
                self.match(self.peek())
                self.skip_white()
                val += self.eval_term()
            elif self.peek() == '-':
                self.match(self.peek())
                self.skip_white()
                val -= self.eval_term()
            else:
                break
        return val

    def eval_term(self):
        val = self.eval_factor() 
        while not self.at_end():
            if self.peek() == ' ':
                self.skip_white()
                if self.peek() not in '-+/)': 
                    val *= self.eval_factor()
                    continue
            self.skip_white()
            if self.peek() == '/':
                self.match(self.peek())
                self.skip_white()
                val /= self.eval_factor()
            else:
                break
            
        return val

    def eval_factor(self):
        # NOTE: no whitespace allowed between these combinators!
        val = self.eval_atom()
        if not self.at_end():
            if self.peek()=='!': 
                self.match(self.peek())
                val = factorial(val) 
            elif self.peek()==':':
                self.match(self.peek())
                nb_samples = self.eval_atom() 
                val = choose(val, nb_samples)
            elif self.peek()=='^':
                self.match(self.peek())
                power = self.eval_atom() 
                val = val**power
        pre(self.at_end() or self.peek() not in '!:^',
            'cannot stack combinatorial operators!')
        return val

    def eval_atom(self):
        # TODO: varnames!
        if self.peek()=='(':
            self.match('(')
            self.skip_white()
            val = self.eval_expr()
            self.skip_white()
            self.match(')')
            return val
        elif self.peek() in '0123456789.':
            digits = '' 
            while (not self.at_end()) and self.peek() in '0123456789.':
                digits += self.peek()
                self.match(self.peek())
            return (float(digits) if '.' in digits else int(digits)) 
        elif self.peek().lower() in 'abcdefghijklmnopqrstuvwxyz':
            name = '' 
            while (not self.at_end()) and self.peek().lower() in 'abcdefghijklmnopqrstuvwxyz':
                name += self.peek()
                self.match(self.peek())
            return self.vals_by_name[name]

if __name__=='__main__':
    vals_by_name = {'T':4, 'eta':np.array([5.0, 6.0])}
    for k, v in vals_by_name.items():
        print('{} = {}'.format(k, str(v)))
    while True:
        print()
        MP = MathParser(input(), vals_by_name)
        print(MP.eval_expr())
