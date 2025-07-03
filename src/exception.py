import sys

def error_message_detail(error,error_detail:sys): # type: ignore
    _,_,exc_tb=error_detail.exc_info()
    filename=exc_tb.tb_frame.f_code.co_filename # type: ignore
    line_number=exc_tb.tb_lineno # type: ignore
    error_message=f"Error occurred in script: [{filename}] line number: [{line_number}] error message: [{str(error)}]"
    return error_message


class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys): # type: ignore
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail)

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return CustomException.__name__.str() + self.error_message # type: ignore
    


        