class AccessControl:
    def __init__(self, authorized_password):
        """
        Initializer for the access control functionality
        :param authorized_password: The correct password a user must input for access
        """
        self.authorized_password = authorized_password.upper()
        self.input_password = ""

    def build_input_password(self, sign):
        """
        Combine the signs in order to create the users input password 
        :param sign: The sign to combine with the rest of the input password so far
        """
        self.input_password += sign

    def is_authorized(self) -> bool:
        """
        Check if the detected sign is authorized for access
        return: A boolean of wheter the 
        """
        return self.input_password == self.authorized_password

    def grant_access(self):
        """
        Function that allows access to be granted to the user
        In deeper integration, this function can be used with hardware integration,
        or PC access
        """
        print(f"\n[ACCESS GRANTED] Authorized password detected: {self.input_password}")

    def deny_access(self):
        """
        Function that denies access to be granted to the user
        In deeper integration, this function can be used with hardware integration or PC access
        """
        print(f"\n[ACCESS DENIED] Unauthorized password: {self.input_password}")
    
    def clear_current_input(self):
        """
        Erase the input password that the user has inputted
        """
        self.input_password = ""
