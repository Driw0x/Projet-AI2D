class Evolution_code:
    # Les informations qui seront retournées
    def __init__(self):
        self.premiere_boucle_for = -1
        self.premiere_struct_if = -1
        self.premiere_boucle_while = -1

        self.range_modif_for = []
        self.range_modif_if = []
        self.range_modif_while = []
        self.range_modif_hors_struct = []

    def __repr__(self):
        s = ""
        s += "premiere_boucle_for = " + str(self.premiere_boucle_for) + "\n"
        s += "premiere_struct_if = " + str(self.premiere_struct_if) + "\n"
        s += "premiere_boucle_while = " + str(self.premiere_boucle_while) + "\n"
        s += "range_modif_for = " + str(self.range_modif_for) + "\n"
        s += "range_modif_if = " + str(self.range_modif_if) + "\n"
        s += "range_modif_while = " + str(self.range_modif_while) + "\n"
        s += "range_modif_hors_struct = " + str(self.range_modif_hors_struct)

        return s