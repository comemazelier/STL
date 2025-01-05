import marimo

__generated_with = "0.9.17"
app = marimo.App(width="medium")


@app.cell
def __():
    # Python Standard Library
    import json

    # Marimo
    import marimo as mo

    # Third-Party Librairies
    import numpy as np
    import matplotlib.pyplot as plt
    import mpl3d
    from mpl3d import glm
    from mpl3d.mesh import Mesh
    from mpl3d.camera import Camera

    import meshio

    np.seterr(over="ignore")  # 🩹 deal with a meshio false warning

    import sdf
    from sdf import sphere, box, cylinder
    from sdf import X, Y, Z
    from sdf import intersection, union, orient, difference
    return (
        Camera,
        Mesh,
        X,
        Y,
        Z,
        box,
        cylinder,
        difference,
        glm,
        intersection,
        json,
        meshio,
        mo,
        mpl3d,
        np,
        orient,
        plt,
        sdf,
        sphere,
        union,
    )


@app.cell
def __(Camera, Mesh, glm, meshio, mo, plt):
    def show(
        filename,
        theta=0.0,
        phi=0.0,
        scale=1.0,
        colormap="viridis",
        edgecolors=(0, 0, 0, 0.25),
        figsize=(6, 6),
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1], xlim=[-1, +1], ylim=[-1, +1], aspect=1)
        ax.axis("off")
        camera = Camera("ortho", theta=theta, phi=phi, scale=scale)
        mesh = meshio.read(filename)
        vertices = glm.fit_unit_cube(mesh.points)
        faces = mesh.cells[0].data
        vertices = glm.fit_unit_cube(vertices)
        mesh = Mesh(
            ax,
            camera.transform,
            vertices,
            faces,
            cmap=plt.get_cmap(colormap),
            edgecolors=edgecolors,
        )
        return mo.center(fig)
    return (show,)


@app.cell
def __(mo, show):
    mo.show_code(show("data/teapot.stl", theta=45.0, phi=30.0, scale=2))
    return


@app.cell
def __(show):
    show("data/cube.stl", theta=45.0, phi=35.0, scale=1)
    return


@app.cell
def __(np):
    def normal(triangles):
        t1, t2, t3 = triangles[:,0,:],triangles[:,1,:],triangles[:,2,:]
        normals=np.cross(t2-t1, t3-t1)
        norme = np.linalg.norm(normals, axis=1, keepdims=True)
        return(normals/norme)
    return (normal,)


@app.cell
def __(normal):
    def make_STL(triangles, normals=None, name=""):
        if normals == None:
            normals = normal(triangles)
        with open(file=f"data/{name}.stl", mode="w") as file:

            file.write(f'solid {name} \n')
            for triangle, normale in zip(triangles, normals):

                file.write(f'  facet normal {normale[0]} {normale[1]} {normale[2]}\n')
                file.write(f'    outer loop\n')

                for point in triangle:
                    file.write(f'      vertex {point[0]} {point[1]} {point[2]}\n')
                file.write('    endloop\n')
                file.write('  endfacet\n')
            file.write(f'endsolid {name}')
    return (make_STL,)


@app.cell
def __(make_STL, np):
    square_triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )

    make_STL(square_triangles, normals=None, name="carre")
    return (square_triangles,)


@app.cell
def __(np):
    def tokenize(stl):
        l = []
        for elt in stl.split('\n'):
            for caractere in elt.split():
                try:
                    l.append(np.float32(caractere))
                except ValueError:
                    l.append(caractere)
        return l
    return (tokenize,)


@app.cell
def __(np, tokenize):
    with open("data/square.stl", mode="rt", encoding="us-ascii") as square_file:
        square_stl = square_file.read()
    tokens = tokenize(square_stl)

    tokens == ['solid', 'square', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(0.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'endloop', 'endfacet', 'facet', 'normal', np.float32(0.0), np.float32(0.0), np.float32(1.0), 'outer', 'loop', 'vertex', np.float32(1.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(0.0), np.float32(1.0), np.float32(0.0), 'vertex', np.float32(1.0), np.float32(0.0), np.float32(0.0), 'endloop', 'endfacet', 'endsolid', 'square']
    return square_file, square_stl, tokens


@app.cell
def __(np):
    def parse(tokens):
        name = tokens[-1]
        triangles = []
        normals = []
        points = [elt for elt in tokens if type(elt) == np.float32]
        n = len(points)
        for i in range(n//(4*3)):
            p0,p1,p2,p3 = [],[],[],[]
            for j in range(3):
                p0.append(points[12*i+j])
                p1.append(points[12*i+j+3])
                p2.append(points[12*i+j+6])
                p3.append(points[12*i+j+9])
            normals.append(p0)
            triangles.append([p1,p2,p3])
        return(np.array(triangles),np.array(normals),name)
    return (parse,)


@app.cell
def __(np, parse, tokenize):
    def verifies_positive_octant_rule(filename):
        # on suppose que file est le nom du fichier STL
        with open(f"data/{filename}.stl", mode='r') as file:
            stl = file.read()
            tokens = tokenize(stl)
            triangles, normals, name= parse(tokens)

            return np.all(triangles >= 0), (np.sum(~(triangles >= 0))/len(triangles))*100
    return (verifies_positive_octant_rule,)


@app.cell
def __(verifies_positive_octant_rule):
    verifies_positive_octant_rule('cube')
    return


@app.cell
def __(normal, np, parse, tokenize):
    def verifies_orientation_rule(filename):
        with open(f"data/{filename}.stl",mode='r') as file:
            stl = file.read()
            tokens = tokenize(stl)
            triangles, normals, name = parse(tokens)
            return np.all(normal(triangles) == normals), (np.sum(~(normal(triangles) == normals))/len(triangles))*100
    return (verifies_orientation_rule,)


@app.cell
def __(verifies_orientation_rule):
    verifies_orientation_rule('cube')
    return


@app.cell
def __(np, parse, tokenize):
    def verifies_ascending_rule(filename):
        with open(f"data/{filename}.stl",mode='r') as file:
            stl = file.read()
            tokens = tokenize(stl)
            triangles, normals, name = parse(tokens)

            z_coordinates = triangles[:,:,2]

            first_column_is_ascending = np.all(z_coordinates[:,0][:-1] <= z_coordinates[:,0][1:])
            total1  = np.sum(~(z_coordinates[:,0][:-1] <= z_coordinates[:,0][1:]))

            second_column_is_ascending = np.all(z_coordinates[:,1][:-1] <= z_coordinates[:,1][1:])
            total2  = np.sum(~(z_coordinates[:,1][:-1] <= z_coordinates[:,1][1:]))

            third_column_is_ascending = np.all(z_coordinates[:,2][:-1] <= z_coordinates[:,2][1:])
            total3  = np.sum(~(z_coordinates[:,2][:-1] <= z_coordinates[:,2][1:]))

            total = total1 + total2 + total3
            erreur = total/z_coordinates.size        
            return (first_column_is_ascending & second_column_is_ascending & third_column_is_ascending), erreur*100
    return (verifies_ascending_rule,)


@app.cell
def __(verifies_ascending_rule):
    verifies_ascending_rule('cube')
    return


@app.cell
def __(parse, tokenize):
    def verifies_shared_edge_rule(filename):
        #complexité ridicule et on abandonne l'idée de donner un pourcentage 
        with open(f"data/{filename}.stl",mode='r') as file:
            stl = file.read()
            tokens = tokenize(stl)
            triangles, normals, name = parse(tokens)

            for triangle in triangles:
                compteur1 = -1 #parce qu'on va recompter l'arête du triangle par la suite
                compteur2 = -1
                compteur3 = -1
                for triangle_test in triangles:
                    if (triangle[0] in triangle_test) & (triangle[1] in triangle_test):
                        compteur1 += 1
                    if (triangle[1] in triangle_test) & (triangle[2] in triangle_test):
                        compteur2 += 1
                    if (triangle[2] in triangle_test) & (triangle[0] in triangle_test):
                        compteur3 += 1
                if (compteur1 > 1) or (compteur2 > 1) or (compteur3 > 1):
                    return False
                if (compteur1==0) or (compteur2==0) or (compteur3==0):
                    return False
            return True
    return (verifies_shared_edge_rule,)


@app.cell
def __(verifies_shared_edge_rule):
    verifies_shared_edge_rule('cube')
    return


@app.cell
def __(make_STL, np):
    def OBJ_to_STL(filename, vertex_count=None, face_count=None, name=''):
        with open(f'data/{filename}.obj','r') as obj_file:
            vertices = []
            faces = []
            if (vertex_count == None) | (face_count == None):
                for line in obj_file:
                    if not line.strip().startswith('#'):
                        if line.strip().startswith('v'):
                            vertices.append((np.array(line.strip().split()[1:])).astype('float'))
                        elif line.strip().startswith('f'):
                            faces.append((np.array(line.strip().split()[1:])).astype('int'))
                vertices = np.array(vertices)
                faces = np.array(faces)

                triangles = vertices[faces - [1]]

                return make_STL(triangles, name=name)
            else:
                pass # on aurait pu essayer d'implémenter une méthode plus efficace en complexité temporelle qui permettrait à l'utilisateur de préciser(en ayant lu les commentaires en début de fichier) le nombre de lignes qui décrivent les vertices et les faces.
    return (OBJ_to_STL,)


@app.cell
def __(OBJ_to_STL):
    OBJ_to_STL('bunny', name='bunny')
    return


@app.cell
def __(show):
    show('data/bunny.stl', theta=14,phi=12,scale=1)
    return


@app.cell
def __(make_STL, np):
    def STL_binary_to_text(stl_filename_in, stl_filename_out):
        with open(stl_filename_in, mode="rb") as file:
            _ = file.read(80)
            n = np.fromfile(file, dtype=np.uint32, count=1)[0]
            normals = []
            faces = []
            for i in range(n):
                normals.append(np.fromfile(file, dtype=np.float32, count=3))
                faces.append(np.fromfile(file, dtype=np.float32, count=9).reshape(3, 3))
                _ = file.read(2)
        stl_text = make_STL(faces, normals, name=stl_filename_out)
    return (STL_binary_to_text,)


@app.cell
def __(STL_binary_to_text):
    STL_binary_to_text('data/dragon.stl', 'dragon_bis')
    return


@app.cell
def __(box, cylinder, difference, intersection, orient, sphere, union):
    def jcad_to_stl(filename):
        with open(f'data/{filename}.jcad','r') as jcad_file:
            correction = jcad_file.read().replace('false','False').replace('true','True')
            dict = eval(correction)
            objects = dict['objects']

            #on suppose que les formes du fichier jcad sont créées dans l'ordre
            shapes = {}
            operations = []
            for elt in objects:
                shape = elt['shape']
                if shape.startswith('Part::Sphere'):
                    position = tuple(elt['parameters']['Placement']['Position'])
                    radius = elt['parameters']['Radius']
                    shapes[elt['name']] = sphere(radius,position)

                elif shape.startswith('Part::Box'):
                    h,l,w = elt['parameters']['Height'], elt['parameters']['Length'], elt['parameters']['Width']
                    #position = tuple(elt['parameters']['Placement']['Position']) En fait, position avec jcad n'est pas identique à center de sdf.
                    shapes[elt['name']] = box((h,l,w)) #,center=position) #il y a un souci si h,l,w ne sont pas sur x,y,z respectivement...Mais pour le fichier à convertir, ça va marcher quand même

                elif shape.startswith('Part::Cylinder'):
                    radius = elt['parameters']['Radius']
                    #position = elt['parameters']['Placement']['Position']
                    axis = elt['parameters']['Placement']['Axis']
                    shapes[elt['name']] = orient(cylinder(radius), axis=axis)

                elif shape.startswith('Part::MultiCommon'):
                    shape1 = elt['dependencies'][0]
                    shape2 = elt['dependencies'][1]
                    operations.append((elt['name'],shape1,shape2))
                    shapes[elt['name']] = intersection(shapes[shape1], shapes[shape2])

                elif shape.startswith('Part::MultiFuse'):
                    shape1 = elt['dependencies'][0]
                    shape2 = elt['dependencies'][1]
                    operations.append((elt['name'],shape1,shape2))
                    shapes[elt['name']] = union(shapes[shape1], shapes[shape2])

                elif shape.startswith('Part::Cut'):
                    shape1 = elt['dependencies'][0]
                    shape2 = elt['dependencies'][1]
                    operations.append((elt['name'],shape1,shape2))
                    shapes[elt['name']] = difference(shapes[shape1], shapes[shape2])

            name_of_last_operation = operations[-1][0]

            shapes[name_of_last_operation].save(f'output/{filename}.stl', step=0.05)
    return (jcad_to_stl,)


@app.cell
def __(jcad_to_stl):
    jcad_to_stl('demo_jcad')
    return


if __name__ == "__main__":
    app.run()
