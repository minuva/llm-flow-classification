from pydantic import BaseModel


class TaskInput(BaseModel):
    llm_input: str
    llm_output: str
