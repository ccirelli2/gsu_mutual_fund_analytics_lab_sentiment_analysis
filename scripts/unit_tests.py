###############################################################################
# Import Libraries
###############################################################################
import unittest
import functions_irregular_tokens as m_irreg_toks
import logging

###############################################################################
# Library Settings 
###############################################################################
logging.basicConfig(level=logging.INFO)


###############################################################################
# Irregular Token Unit Tests
###############################################################################

class TestConditionResult(unittest.TestCase):

    def test_positive_condition_n_confirm_param(self):
        text="""today is a good day to code""".split(' ')
        t_confirm=m_irreg_toks.irregular_token_conditions(
                irregular_token='day', sent_tokens=text,
                window_direction='both', window_size=2, anchor_word='day',
                targets=['code'], token_condition='confirm')
        self.assertEqual(t_confirm, False)

    def test_positive_condition_n_reverse_param(self):
        text="""today is a good day to code""".split(' ')
        t_confirm=m_irreg_toks.irregular_token_conditions(
                irregular_token='day', sent_tokens=text,
                window_direction='both', window_size=2, anchor_word='day',
                targets=['code'], token_condition='reverse')
        self.assertEqual(t_confirm, True)




if __name__=="__main__":
    unittest.main()
















