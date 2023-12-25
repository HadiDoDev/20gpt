from typing import Optional, Any

import pymongo
import uuid
from datetime import datetime

import configs


class Database:
    def __init__(self, mongodb_uri=None):
        if mongodb_uri:
            self.client =  pymongo.MongoClient(mongodb_uri)
        else:
            self.client = pymongo.MongoClient(configs.mongodb_uri)
        self.db = self.client["20gpt_bot"]

        self.user_collection = self.db["user"]
        self.dialog_collection = self.db["dialog"]

    def check_if_user_exists(self, user_id: int, raise_exception: bool = False):
        if self.user_collection.count_documents({"_id": user_id}) > 0:
            return True
        else:
            if raise_exception:
                raise ValueError(f"User {user_id} does not exist")
            else:
                return False

    def add_new_user(
        self,
        user_id: int,
        chat_id: int,
        username: str = "",
        first_name: str = "",
        last_name: str = "",
    ):
        user_dict = {
            "_id": user_id,
            "chat_id": chat_id,

            "username": username,
            "first_name": first_name,
            "last_name": last_name,

            "last_interaction": datetime.now(),
            "first_seen": datetime.now(),

            "current_dialog_id": None,
            "current_chat_mode": "assistant",
            "current_model": configs.models["available_text_models"][0],

            "n_used_tokens": {},
            
            "credit": {
                "is_trial": True,
                "used_rials": 0.0,
                "total_rials": 50000.0,
                "chat_modes": [],
                "increased_at": None,
                "decreased_at": None,
                "latest_chat_modes_added": []
            },
            
            "n_generated_images": 0,
            "n_transcribed_seconds": 0.0  # voice message transcription
        }

        if not self.check_if_user_exists(user_id):
            self.user_collection.insert_one(user_dict)

    def start_new_dialog(self, user_id: int):
        self.check_if_user_exists(user_id, raise_exception=True)

        dialog_id = str(uuid.uuid4())
        dialog_dict = {
            "_id": dialog_id,
            "user_id": user_id,
            "chat_mode": self.get_user_attribute(user_id, "current_chat_mode"),
            "start_time": datetime.now(),
            "model": self.get_user_attribute(user_id, "current_model"),
            "messages": []
        }

        # add new dialog
        self.dialog_collection.insert_one(dialog_dict)

        # update user's current dialog
        self.user_collection.update_one(
            {"_id": user_id},
            {"$set": {"current_dialog_id": dialog_id}}
        )

        return dialog_id

    def get_user_attribute(self, user_id: int, key: str):
        self.check_if_user_exists(user_id, raise_exception=True)
        user_dict = self.user_collection.find_one({"_id": user_id})

        if key not in user_dict:
            return None

        return user_dict[key]

    def set_user_attribute(self, user_id: int, key: str, value: Any):
        self.check_if_user_exists(user_id, raise_exception=True)
        self.user_collection.update_one({"_id": user_id}, {"$set": {key: value}})

    def update_n_used_tokens(self, user_id: int, model: str, n_input_tokens: int, n_output_tokens: int):
        n_used_tokens_dict = self.get_user_attribute(user_id, "n_used_tokens")

        if model in n_used_tokens_dict:
            n_used_tokens_dict[model]["n_input_tokens"] += n_input_tokens
            n_used_tokens_dict[model]["n_output_tokens"] += n_output_tokens
        else:
            n_used_tokens_dict[model] = {
                "n_input_tokens": n_input_tokens,
                "n_output_tokens": n_output_tokens
            }

        self.set_user_attribute(user_id, "n_used_tokens", n_used_tokens_dict)

    def get_dialog_messages(self, user_id: int, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id")

        dialog_dict = self.dialog_collection.find_one({"_id": dialog_id, "user_id": user_id})
        return dialog_dict["messages"]

    def set_dialog_messages(self, user_id: int, dialog_messages: list, dialog_id: Optional[str] = None):
        self.check_if_user_exists(user_id, raise_exception=True)

        if dialog_id is None:
            dialog_id = self.get_user_attribute(user_id, "current_dialog_id")

        self.dialog_collection.update_one(
            {"_id": dialog_id, "user_id": user_id},
            {"$set": {"messages": dialog_messages}}
        )

    def check_if_user_has_credit(self, user_id: int, chat_mode : str, raise_exception: bool = True):
        credit = self.get_user_attribute(user_id, "credit")
        
        has_credit = credit['total_rials'] > credit['used_rials']

        if credit['is_trial'] and has_credit:
            return True
        elif has_credit and chat_mode in credit['chat_modes']:
            return True
        else:
            if raise_exception:
                raise ValueError(f"\nشما اعتبار کافی برای درس انتخاب شده را برای ادامه فرایند ندارید.\n لطفا درس انتخاب شده را تغییر داده یا با استفاده از منو و گزینه /purchase اقدام به خرید بسته در دروس مورد نظر خود و افزایش اعتبار کنید.")
            else:
                return False

    def decrease_user_credit(self, user_id: int, n_used_rials: float):
        user_credit = self.get_user_attribute(user_id, "credit")
        user_credit['used_rials'] += n_used_rials
        user_credit['decreased_at'] = datetime.now()

        self.set_user_attribute(user_id, "credit", user_credit)

    def increase_user_credit(self, user_id: int, n_total_rials: float = None, chat_modes: list = None):
        user_credit = self.get_user_attribute(user_id, "credit")

        if n_total_rials:
            if user_credit['is_trial']:
                user_credit['total_rials'] = 0.0
                user_credit['is_trial'] = False

            user_credit['increased_at'] = datetime.now()
            user_credit['total_rials'] += n_total_rials
        
        if chat_modes:
            if not all(chat_mode in list(configs.chat_modes.keys()) for chat_mode in chat_modes):
                raise ValueError(f"Invalid chat modes. {chat_modes}")
            
            user_credit['chat_modes'].extend(chat_modes)
            user_credit['latest_chat_modes_added'].extend(chat_modes)
        
        if chat_modes or n_total_rials:
                
            self.set_user_attribute(user_id, "credit", user_credit)
