


/*
 ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 2000 */


// @sect3{Include files}

#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>

#include <fstream>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>


//(<code> )

using namespace dealii;


// @sect3{The <code>Step6</code> class template}

// We have added two new functions to the main class: (<code> set_all_subdomain_objects) and (<code> get_fe_function).
// Since we are creating new objects for each subdomain including triangulation, system_rhs, solution, etc., we have
// to explicitly make each aware of the other, which is exactly what is achieved by (<code> set_all_subdomain_objects).
// A new object that helps with this is the vector where these subdomain objects are stored,
// (<code> subdomain_objects).
// Meanwhile, (<code> get_fe_function) provides a way to retrieve the appropriate function to apply as a boundary
// condition on a particular boundary_id. Helpful to this function are the vectors (<code> solutionfunction_vector) and
// (<code> solution_vector) which are populated at the end of (<code> solve). More details are provided there.


template <int dim>
class Step6
{
public:
    Step6(const unsigned int subdomain);
    void run(const unsigned int cycle, const unsigned int s, std::string method);

    void set_all_subdomain_objects (const std::vector<std::shared_ptr<Step6<dim>>> &objects)
    { subdomain_objects = objects; }

    std::vector<Vector<double>> solution_vector;
    std::vector<std::unique_ptr<Functions::FEFieldFunction<dim>>> solutionfunction_vector;


private:
    void setup_system();
    void assemble_system(const unsigned int s, std::string method);
    Functions::FEFieldFunction<dim> & get_fe_function(const unsigned int boundary_id,
                                                      const unsigned int s,
                                                      std::string method);
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle, const unsigned int s, std::string method) const;

    std::vector<std::shared_ptr<Step6<dim>>> subdomain_objects;

    const unsigned int subdomain;

    Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;
    SparseMatrix<double> system_matrix;
    SparsityPattern      sparsity_pattern;

    Vector<double> solution;
    Vector<double> system_rhs;

};


// @sect4{Nonconstant coefficients}


template <int dim>
double coefficient(const Point<dim> &p)
{
    if (p.square() < 0.5 * 0.5)
        return 20;
    else
        return 1;
}



// @sect3{The <code>Step6</code> class implementation}


// @sect4{Step6::Step6}

// Here we have a few additions. First is the initialization of the subdomain on which we are preparing to solve.


template <int dim>
Step6<dim>::Step6(const unsigned int subdomain)

        : subdomain (subdomain),
        fe(2),
        dof_handler(triangulation)

{

// Then we manually create the subdomains using their defining corner points:

    const std::vector<Point<2>> corner_points = {Point<2>(-0.875, -0.125), Point<2>(0.125, 0.875),
                                                 Point<2>(-0.125, -0.125), Point<2>(0.875, 0.875),
                                                 Point<2>(-0.125,-0.875), Point<2>(0.875,0.125),
                                                 Point<2>(-0.875,-0.875), Point<2>(0.125,0.125)};



    GridGenerator::hyper_rectangle(triangulation, corner_points[2 * subdomain],
                                   corner_points[2 * subdomain + 1]);

    triangulation.refine_global(2);



// Lastly, we set boundary_ids (those not explicitly set to a value are zero by default, a fact that we use later):


    //Set the boundary_ids of edges of subdomain0
    if (subdomain == 0) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Right edge:

                //Set boundary_id to 2 along the portion of gamma2 that makes up part of the right
                // boundary edge of subdomain0
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) >= 0.125))
                    face->set_boundary_id(2);

                //Set boundary_id to 6 along gamma6, the remaining portion of subdomain0's right edge
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) < 0.125))
                    face->set_boundary_id(6);

                //Bottom edge:

                //Set boundary_id to 4 along the portion of gamma4 that makes up part of the bottom
                // boundary edge of subdomain0
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) < -0.125))
                    face->set_boundary_id(4);

                //Set boundary_id to 8 along gamma8, the remaining portion of subdomain0's bottom edge
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) >= -0.125))
                    face->set_boundary_id(8);

                //Remaining edges have boundary_ids of 0 by default.

            }

    //Set the boundary_ids of edges of subdomain1
    } else if (subdomain == 1) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Left edge:

                //Set boundary_id to 1 along portion of gamma1 that makes up part of the left
                // boundary edge of subdomain1
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) >= 0.125))
                    face->set_boundary_id(1);

                //Set boundary_id to 5 along gamma5, the remaining portion of subdomain1's left edge
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) < 0.125))
                    face->set_boundary_id(5);

                //Bottom edge:

                //Set boundary_id to 4 along portion of gamma4 that makes up part of the bottom
                // boundary edge of subdomain1
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) >= 0.125))
                    face->set_boundary_id(4);

                //Set boundary_id to 8 along gamma8, the remaining portion of subdomain1's bottom edge
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) < 0.125))
                    face->set_boundary_id(8);

                //Remaining edges have boundary_ids of 0 by default.

            }

    //Set the boundary_ids of edges of subdomain2
    } else if (subdomain == 2) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Left edge:

                //Set boundary_id to 1 along portion of gamma1 that makes up part of the left
                // boundary edge of subdomain2
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) <= -0.125))
                    face->set_boundary_id(1);

                //Set boundary_id to 5 along gamma5, the remaining portion of subdomain2's left edge
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) > -0.125))
                    face->set_boundary_id(5);

                //Top edge:

                //Set boundary_id to 3 along portion of gamma3 that makes up part of the top
                // boundary edge of subdomain2
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) >= 0.125))
                    face->set_boundary_id(3);

                //Set boundary_id to 7 along gamma7, the remaining portion of subdomain2's top edge
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) < 0.125))
                    face->set_boundary_id(7);

                //Remaining edges have boundary_ids of 0 by default.

            }

    //Set the boundary_ids of edges of subdomain3
    } else if (subdomain == 3) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Right edge:

                //Set boundary_id to 2 along portion of gamma2 that makes up part of the right
                // boundary edge of subdomain3
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) <= -0.125))
                    face->set_boundary_id(2);

                //Set boundary_id to 6 along gamma6, the remaining portion of subdomain3's right edge
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) > -0.125))
                    face->set_boundary_id(6);

                //Top edge:

                //Set boundary_id to 3 along portion of gamma3 that makes up part of the top
                // boundary edge of subdomain3
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) <= -0.125))
                    face->set_boundary_id(3);

                //Set boundary_id to 7 along gamma7, the remaining portion of subdomain3's top edge
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) > -0.125))
                    face->set_boundary_id(7);

                //Remaining edges have boundary_ids of 0 by default.

            }

    } else
        Assert (false, ExcInternalError());


    // Setting the boundary_ids as above is clearly a lengthy, tedious, and bug-prone process that makes the program,
    // as it currently stands, combersome to use. This process will eventually be replaced this with an automated
    // method, but I needed a highly controlled system running before playing with this feature.


    setup_system();

    // (<code> setup_system) makes (<code> solution) of the right size, (<code> n_dofs), and distributes dofs.
    // So after (<code> setup_system), we can make a FEFieldFunction (using solution and dof_handler) equivalent
    // to the zero function (because solution is initialized with a value of zero by default) and make this
    // FEFieldfunction the first entry of each solutionfunction_vector:

    std::unique_ptr<Functions::FEFieldFunction<dim>> zero_function =
            std::make_unique<Functions::FEFieldFunction<dim>>(dof_handler, solution);
    solutionfunction_vector.emplace_back (std::move(zero_function));

}


// @sect4{Step6::setup_system}

// This function remains identical to the original.

template <int dim>
void Step6<dim>::setup_system()
{
    dof_handler.distribute_dofs(fe);
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             constraints);

    constraints.close();

    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ false);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);


}


// @sect4{Step6::get_fe_function}

// As previously mentioned, an additional function is needed to get the appropriate solution for each edge and apply it
// as an (<code> FEFieldFunction) boundary condition for the subdomain bounded by the given edge. This association
// between subdomains, which subdomain solution is imposed as a boundary condition to which edge, is also done
// explicitly here, but will be replaced with an automated system.

template<int dim>
Functions::FEFieldFunction<dim> &
        Step6<dim>::get_fe_function(unsigned int boundary_id, unsigned int s, std::string method)
{

    // Get other subdomain id
    types::subdomain_id relevant_subdomain;


    // Boundary conditions for multiplicative and additive Schwarz differ, so we need to check which method the
    // user would like to use:

    // @sect5{Multiplicative Schwarz}

    if (method == "Multiplicative") {


        if (s == 0) {
            if (boundary_id == 2)
                relevant_subdomain = 1;

            else if (boundary_id == 6)
                relevant_subdomain = 2;

            else if (boundary_id == 8)
                relevant_subdomain = 3;

            else if (boundary_id == 4)
                relevant_subdomain = 3;

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else if (s == 1) {
            if (boundary_id == 1)
                relevant_subdomain = 0;

            else if (boundary_id == 5)
                relevant_subdomain = 0;

            else if (boundary_id == 8)
                relevant_subdomain = 3;

            else if (boundary_id == 4)
                relevant_subdomain = 2;

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else if (s == 2) {

            if (boundary_id == 3)
                relevant_subdomain = 1;

            else if (boundary_id == 5)
                relevant_subdomain = 0;

            else if (boundary_id == 7)
                relevant_subdomain = 1;

            else if (boundary_id == 1)
                relevant_subdomain = 3;

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else if (s == 3) {

            if (boundary_id == 3)
                relevant_subdomain = 0;

            else if (boundary_id == 6)
                relevant_subdomain = 2;

            else if (boundary_id == 7)
                relevant_subdomain = 1;

            else if (boundary_id == 2)
                relevant_subdomain = 2;

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else Assert (false, ExcInternalError());


        std::cout << "              The BC function for subdomain " << s << " on gamma" << boundary_id <<
                  " is from subdomain " << relevant_subdomain << std::endl;

        //For Multiplicative Schwarz, we impose the most recently computed solution from neighboring subdomains as the
        // BC of the current subdomain, so we retrieve the last entry of the appropriate solutionfunction_vector:

        return *subdomain_objects[relevant_subdomain]->solutionfunction_vector.back();


    //ADDITIVE//--------------------------------------------------------------------------------------------------------







    // @sect5{Additive Schwarz}

    } else if (method == "Additive") {

    // For Additive Schwarz, we impose as a boundary condition a linear combination of the solutions from
    // subdomains overlapping at any given edge. These solutions are all from the previous cycle, so we retrieve
    // the second to last entry of (<code>solution_vector) if the solution has already been computed on the
    // (<code>relevant_subdomain) in the current cycle and retrieve the last entry of (<code>relevant_subdomain)'s
    // (<code>solution_vector) if the solution on the relevant_subdomain has not yet been computed in the current cycle.
    // We know that, if we are solving on subdomain_i, then subdomain_k for k<i have been solved in the current cycle
    // while subdomain_k for k>i have not been solved in the current cycle.
    // We also must consider the fact that the second to last entry of solutionfunction_vector may not exist; namely
    // when the size of solutionfunction_vector is 1. In this case, we need to retrieve its only entry, which happens
    // to be the vector's last entry. We can retrieve this as before (with .back()).
    // Because we must impose a linear combination of these solutions as a boundary condition, we use their
    // (<code>solution_vectors) rather than their (<code>solutionfunction_vectors) because (<code>solution_vector) is
    // a vector of Vectors, which is easier to algebraically manipulate than a vector of FEFieldFunctions.


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Inaccurate implementation: BC entirely retrieved from neighboring subdomain rather than being a weighted average
    // of all solutions from subdomains in the overlapping region:
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //if ((subdomain_objects[relevant_subdomain] -> solutionfunction_vector.size()) == 1)
        //    return *subdomain_objects[relevant_subdomain]->solutionfunction_vector.back();



        //else {// subdomain_objects[relevant_subdomain] -> solutionfunction_vector.size() > 1
        //    if (relevant_subdomain > s) //then solution for relevant_subdomain has not been computed in the current cycle,
                // so retrieving the last element of its solutionfunction_vector is retrieving its
                // solution from the previous cycle
        //        return *subdomain_objects[relevant_subdomain]->solutionfunction_vector.back();

        //    else if (relevant_subdomain < s) { //then solution for relevant_subdomain has been computed in the current cycle,
                // so retrieving the second to last element of its solutionfunction_vector is
                // retrieving its solution from the previous cycle
        //        return *subdomain_objects[relevant_subdomain]->solutionfunction_vector.rbegin()[1];

        //    } else { // relevant_subdomain = current_subdomain
        //        std::cout << "Error: relevant_subdomain cannot be the same as the current subdomain" << std::endl;
        //        Assert (false, ExcInternalError());
        //    }

    // end of inaccurate implementation ////////////////////////////////////////////////////////////////////////////////


    //Incomplete correction below:


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        std::vector<int> relevant_subdomains;

        if (s == 0) {
            if (boundary_id == 2) {
                relevant_subdomains = {1, 0};

            } else if (boundary_id == 6) {
                relevant_subdomains = {2,0,1,3};

            } else if (boundary_id == 8) {
                relevant_subdomains = {3,0,1,2};

            } else if (boundary_id == 4) {
                relevant_subdomains = {3,0,1,2};

            } else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else if (s == 1) {
            if (boundary_id == 1) {
                relevant_subdomains = {0,1};

            } else if (boundary_id == 5) {
                relevant_subdomains = {0,1,2,3};

            } else if (boundary_id == 8) {
                relevant_subdomains = {3,0,1,2};

            } else if (boundary_id == 4) {
                relevant_subdomains = {2,1};

            } else //boundary_id == 0
                Assert (false, ExcInternalError());


        } else if (s == 2) {

            if (boundary_id == 3)
                relevant_subdomains = {1,2};

            else if (boundary_id == 5)
                relevant_subdomains = {0,1,2,3};

            else if (boundary_id == 7)
                relevant_subdomains = {1,0,2,3};

            else if (boundary_id == 1)
                relevant_subdomains = {3,2};

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else if (s == 3) {

            if (boundary_id == 3)
                relevant_subdomains = {0,3};

            else if (boundary_id == 6)
                relevant_subdomains = {2,0,1,3};

            else if (boundary_id == 7)
                relevant_subdomains = {1,0,2,3};

            else if (boundary_id == 2)
                relevant_subdomains = {2,3};

            else //boundary_id == 0
            Assert (false, ExcInternalError());


        } else Assert (false, ExcInternalError());



        std::vector<Vector<double>> overlapping_solutions;

        for (int i=0; i < relevant_subdomains.size(); i++) {

            if ((subdomain_objects[relevant_subdomains[i]]->solution_vector.size()) == 1)
                overlapping_solutions.push_back(subdomain_objects[relevant_subdomains[i]]->solution_vector.back());


            else {// subdomain_objects[relevant_subdomain] -> solution_vector.size() > 1
                if (relevant_subdomain >= s)

                    //then solution for relevant_subdomain has not been computed in the current cycle,
                    // so retrieving the last element of its solution_vector is retrieving its
                    // solution from the previous cycle

                    overlapping_solutions.push_back(subdomain_objects[relevant_subdomains[i]]->solution_vector.back());



                else

                    // Then relevant_subdomain < s and the solution for relevant_subdomain has been computed
                    // in the current cycle, so retrieving the second to last element of its solution_vector is
                    // retrieving its solution from the previous cycle

                    overlapping_solutions.push_back(subdomain_objects[relevant_subdomains[i]]->solution_vector.rbegin()[1]);



            //When solving subdomains in parallel for each cycle, the above if-else is no longer necessary. No matter
            //what, we only need the last entry of the solution_vector belonging to each of the entries of the
            //relevant_subdomains vector.

            }

        }

        //TODO: loop through all overlapping_solutions entries, resulting in an evenly weighted average of all entries
        // of this overlapping_solutions vector.
        Vector<double> solution_in_overlap;
        solution_in_overlap = VectorOperation::add(1/(overlapping_solutions.size()),
                                                  overlapping_solutions[0],
                                                  ...
                                                  1/(overlapping_solutions.size()),
                                                  overlapping_solutions[overlapping_solutions.size()-1]);


        Functions::FEFieldFunction<2> fe_function (dof_handler, solution_in_overlap);
        return fe_function;




    } else { //Neither Multiplicative Schwarz nor Additive Schwarz were chosen as the solving method
        std::cout << "Error: 'Multiplicative' or 'Additive' must be chosen as the solving method" << std::endl;
        Assert (false, ExcInternalError());

    }


}






// @sect4{Step6::assemble_system}

// Next, we actually impose the appropriate solutions as boundary conditions on their appropriate edges, which is
// done with VectorTools::interpolate_boundary_values. We specify the function that will be applied as a boundary
// condition on the nonzero boundary_ids using get_fe_function, but on the boundaries with boudary_id of zero by
// default (the edges on the exterior of the entire domain) we impose the homogenous direchlet boundary condition
// by using Functions::ZeroFunction<dim>() as the appropriate boundary function.


template <int dim>
void Step6<dim>::assemble_system(unsigned int s, std::string method)
{
    system_matrix = 0;
    system_rhs = 0;

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_rhs(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        cell_matrix = 0;
        cell_rhs = 0;
        fe_values.reinit(cell);
        for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
            const double current_coefficient =
                    coefficient<dim>(fe_values.quadrature_point(q_index));
            for (const unsigned int i : fe_values.dof_indices()) {
                for (const unsigned int j : fe_values.dof_indices())
                    cell_matrix(i, j) +=
                            (current_coefficient *              // a(x_q)
                             fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                             fe_values.shape_grad(j, q_index) * // grad phi_j(x_q)
                             fe_values.JxW(q_index));           // dx
                cell_rhs(i) += (1.0 *                               // f(x)
                                fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                fe_values.JxW(q_index));            // dx
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(
                cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }


    std::map<types::global_dof_index, double> boundary_values;

    //Impose boundary condition on edges with boundary_id of 0
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);


    //Impose boundary conditions on edges of subdomain0 with nonzero boundary_ids
    if (s == 0){

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 2,
                                                 get_fe_function(2, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 4,
                                                 get_fe_function(4, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 6,
                                                 get_fe_function(6, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 8,
                                                 get_fe_function(8, s, method),
                                                 boundary_values);


    //Impose boundary conditions on edges of subdomain1 with nonzero boundary_ids
    } else if (s == 1){

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 1,
                                                 get_fe_function(1, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 4,
                                                 get_fe_function(4, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 5,
                                                 get_fe_function(5, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 8,
                                                 get_fe_function(8, s, method),
                                                 boundary_values);

    //Impose boundary conditions on edges of subdomain2 with nonzero boundary_ids
    } else if (s == 2){
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 1,
                                                 get_fe_function(1, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 3,
                                                 get_fe_function(3, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 5,
                                                 get_fe_function(5, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 7,
                                                 get_fe_function(7, s, method),
                                                 boundary_values);

    //Impose boundary conditions on edges of subdomain3 with nonzero boundary_ids
    } else if (s == 3){
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 2,
                                                 get_fe_function(2, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 3,
                                                 get_fe_function(3, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 6,
                                                 get_fe_function(6, s, method),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 7,
                                                 get_fe_function(7, s, method),
                                                 boundary_values);

    } else
    Assert (false, ExcInternalError());


    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);


}


// @sect4{Step6::solve}

// The same solver as before is used because this is not the focus of this modified program.

template <int dim>
void Step6<dim>::solve()
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

// Now we can store solution in vector that can be retrieved elsewhere. As previously mentioned, it will be helpful to
// store the solution as a Vector for algebraic manipulation required in the implementation of additive Schwarz.
    solution_vector.push_back(solution);

// Here we store the solution as function that can be retrieved elsewhere. Namely, this will be used in the
// implementation of multiplicative Schwarz.
    std::unique_ptr<Functions::FEFieldFunction<dim>> solutionfunction_pointer =
            std::make_unique<Functions::FEFieldFunction<dim>>(dof_handler, solution);
    solutionfunction_vector.emplace_back (std::move(solutionfunction_pointer));

    std::cout << "  max solution value=" << solution.linfty_norm()
              << std::endl;

}



// @sect4{Step6::refine_grid}

template <int dim>
void Step6<dim>::refine_grid()
{
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                       solution,
                                       estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);
    triangulation.execute_coarsening_and_refinement();


}



// @sect4{Step6::output_results}

template <int dim>
void Step6<dim>::output_results(const unsigned int cycle, const unsigned int s, std::string method) const
{

    /*if (method == "Multiplicative") {

        {

            GridOut grid_out;


            //if (method == "Additive")
            //    std::ofstream output("ASgrid-" + std::to_string(s*100 + cycle) + "-" + method +".gnuplot");
            //else //(method == "Multiplicative")
            //    std::ofstream output("MSgrid-" + std::to_string(s*100 + cycle) + "-" + method +".gnuplot");


            std::ofstream output("MSgrid-" + std::to_string(s * 100 + cycle) + ".gnuplot");


            GridOutFlags::Gnuplot gnuplot_flags(false, 5);
            grid_out.set_flags(gnuplot_flags);
            MappingQGeneric<dim> mapping(3);
            grid_out.write_gnuplot(triangulation, output, &mapping);

        }

        {

            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(solution, "solution");
            data_out.build_patches();

            std::ofstream output("MSsolution-" + std::to_string(s * 100 + cycle) + "-" + method + ".vtu");

            data_out.write_vtu(output);


        }

    } else { //method == "Additive"

        {

            GridOut grid_out;

            std::ofstream output("ASgrid-" + std::to_string(s * 100 + cycle) + ".gnuplot");


            GridOutFlags::Gnuplot gnuplot_flags(false, 5);
            grid_out.set_flags(gnuplot_flags);
            MappingQGeneric<dim> mapping(3);
            grid_out.write_gnuplot(triangulation, output, &mapping);

        }

        {

            DataOut<dim> data_out;
            data_out.attach_dof_handler(dof_handler);
            data_out.add_data_vector(solution, "solution");
            data_out.build_patches();

            std::ofstream output("ASsolution-" + std::to_string(s * 100 + cycle) + "-" + method + ".vtu");

            data_out.write_vtu(output);


        }


    }*/

    {

        GridOut grid_out;

        std::ofstream output("grid-" + std::to_string(s * 100 + cycle) + ".gnuplot");

        GridOutFlags::Gnuplot gnuplot_flags(false, 5);
        grid_out.set_flags(gnuplot_flags);
        MappingQGeneric<dim> mapping(3);
        grid_out.write_gnuplot(triangulation, output, &mapping);

    }

    {

        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(solution, "solution");
        data_out.build_patches();

        std::ofstream output("solution-" + std::to_string(s * 100 + cycle) + ".vtu");

        data_out.write_vtu(output);

    }



}


// @sect4{Step6::run}

// The (<code> run) function remains nearly identical; some of the functions it calls merely needed extra arguments.

template <int dim>
void Step6<dim>::run(const unsigned int cycle, const unsigned int s, std::string method) {

    std::cout << "Cycle:  " << cycle << std::endl;
    std::cout << "Subdomain:  " << s << std::endl;

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    refine_grid();

    setup_system();

    std::cout << " After calling refine_grid():" << std::endl;
    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;


    assemble_system(s, method);


        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    solve();

    output_results(cycle, s, method);

}




int main()
{
    try
    {

        // We now construct each subdomain problem and store them all in a vector to solve iteratively.

        std::vector<std::shared_ptr<Step6<2>>> subdomain_problems;

        subdomain_problems.push_back (std::make_shared<Step6<2>> (0));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (1));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (2));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (3));


        // Tell each of the objects representing one subdomain each about the objects representing all of the
        // other subdomains:
        for (unsigned int s=0; s<subdomain_problems.size(); ++s) {
            subdomain_problems[s] -> set_all_subdomain_objects(subdomain_problems);
        }

        // Choose whether we use Multiplicative or Additive Schwarz to solve
        std::string method;
        std::cout << "Which Schwarz method would you like to use to solve? (Multiplicative or Additive)" << std::endl;
        std::cin >> method;

        // Now we can actually solve each subdomain problem
        for (unsigned int cycle=1; cycle<10; ++cycle)
            for (unsigned int s=0; s<subdomain_problems.size(); ++s) {
                subdomain_problems[s] -> run(cycle, s, method);
            }

    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    return 0;
}




