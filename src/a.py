class Point():
    
    def __init__(self, name, latitude, longitude):
        """__init__ is the initial file $$a^2 = b^2 + c^2$$ and
        $$ \\sigma = \\alpha + \\beta $$

        $$e^{i\pi} + 1 = 0$$

        $$
        \\begin{align}
            \\label{eqn:eqlabel}
            \\begin{split}
            f(x) &= x^2 ,
            \\newline
            g(x) &= \exp( x ) .
            \\end{split}
            \\end{align}
        $$

        .. image:: ./../download.jpg
        
        .. list-table:: Title
            :widths: 25 25 50
            :header-rows: 1

            * - Heading row 1, column 1
                - Heading row 1, column 2
                - Heading row 1, column 3
            * - Row 1, column 1
                -
                - Row 1, column 3
            * - Row 2, column 1
                - Row 2, column 2
                - Row 2, column 3
        
        Example:
            Examples can be given using `conf.py` either the ``Example`` or ``Examples``
            sections. Sections support any reStructuredText formatting, including
            literal blocks::

                $ python example_google.py
                
                >>> print([i for i in example_generator(4)])
                [0, 1, 2, 3]

        Note:
            Do not include the `self` parameter in the ``Args`` section.
            
            Section breaks are created by resuming unindented text. Section breaks
            are also implicitly created anytime a new section starts.

        Attributes:
            module_level_variable1 (int): Module level variables may be documented in
                either the ``Attributes`` section of the module docstring, or in an
                inline docstring immediately following the variable.

                Either form is acceptable, but the two should not be mixed. Choose
                one convention to document module level variables and be consistent
                with it.

        Todo:
            * For module TODOs
            * You have to also use ``sphinx.ext.todo`` extension

    
        
        Args:
            name ([type]): [description]
            latitude ([type]): [description]
            longitude ([type]): [description]

        Raises:
            TypeError: [description]
            ValueError: [description]
        """
        self.name = name
        
        if not ( type(name) == str):
            raise TypeError("name must be a string")
        
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            raise ValueError("Invalid latitude")
        
        
        self.latitude = latitude
        self.longitude = longitude
    
    def get_lat_long(self):
        return (self.latitude, self.longitude)
    