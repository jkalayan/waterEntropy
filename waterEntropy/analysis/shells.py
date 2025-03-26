"""Store shell information here"""


class Shell:
    """Represent a single atom shell with dynamic properties, which are set in
    the ShellCollection class."""

    def __init__(self, atom_idx):
        object.__setattr__(self, "atom_idx", atom_idx)  # Store atom index directly
        object.__setattr__(self, "properties", {})  # Internal dictionary

    def __getattr__(self, key: str):
        """Allow dot notation access to stored properties.

        :param key: name of the property in the properties dictionary
        """
        if key in self.properties:
            return self.properties[key]
        raise AttributeError(f"Property '{key}' not found for atom {self.atom_idx}")

    def __setattr__(self, key: str, value):
        """Allow adding properties dynamically, except for 'atom_idx'.

        :param key: name of the property being added to the properties dictionary
        :param value: value for the key being added to the properties dictionary
        """
        if key == "atom_idx":  # Ensure atom index remains immutable
            object.__setattr__(self, key, value)
        else:
            self.properties[key] = value  # Store additional properties

    def __repr__(self):
        """Print the atom and its properties."""
        return f"Shell(atom_idx={self.atom_idx}, properties={self.properties})"


class ShellCollection:
    """Manage multiple atom shells and allow adding/updating properties."""

    def __init__(self):
        self.shells = {}  # Dictionary storing Shell objects indexed by atom index

    def add_data(self, atom_idx: int, UA_shell: list[int]):
        """Add a new atom shell to the collection if it doesn't exist and set
        various properties used to describe the class.

        :param atom_idx: the heavy atom index of the central atom in a shell
        :param UA_shell: the list of heavy atom indices in the shell of atom_idx
        """
        if atom_idx not in self.shells:
            self.shells[atom_idx] = Shell(atom_idx)
            self.set_property(atom_idx, "atom_idx", atom_idx)
            self.set_property(atom_idx, "UA_shell", UA_shell)
            self.set_property(atom_idx, "nearest_nonlike_idx", None)
            self.set_property(atom_idx, "labels", None)
            self.set_property(atom_idx, "donates_to_labels", None)
            self.set_property(atom_idx, "accepts_from_labels", None)

    def set_property(self, atom_idx, key: str, value):
        """Set a property for a specific atom shell, ensuring it exists first.

        :param atom_idx: the heavy atom index of the central atom in a shell
        :param key: the name of the key being added to the shells dictionary
        :param value: the value of the key being added to the shells dictionary
        """
        if atom_idx not in self.shells:
            self.shells[atom_idx] = Shell(atom_idx)  # Auto-create atom if missing
        setattr(self.shells[atom_idx], key, value)

    def find_shell(self, atom_idx: int):
        """
        Get the shell instance if atom_idx is a key in shells dictionary

        :param atom_idx: index of central atom in shell
        """
        return self.shells.get(atom_idx, None)

    def __repr__(self):
        """Return a dictionary-like representation of stored atoms."""
        return repr(self.shells)
