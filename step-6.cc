


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
#include <deal.II/base/timer.h>
#include <deal.II/base/table_handler.h> //////////////////////////////////////////////////////////////////////////////// table

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




using namespace dealii;


// @sect3{The (<code>Step6) class template}

// We have added two new functions to the main class: (<code> set_all_subdomain_objects)
// and (<code> get_fe_function). Since we are creating new objects for each subdomain
// including triangulation, system_rhs, solution, etc., we have to explicitly make each
// aware of the other, which is exactly what is achieved by (<code> set_all_subdomain_objects).
// A new object that helps with this is the vector where these subdomain objects are stored,
// (<code> subdomain_objects). Meanwhile, (<code> get_fe_function) provides a way to retrieve
// the appropriate function to apply as a boundary condition for multiplicative Schwarz.
// Helpful to this function is the vector (<code> solutionfunction_vector), which is
// populated at the end of (<code> solve). More details are provided there.


template <int dim>
class Step6
{
public:
    Step6(const unsigned int s);

    void run(const unsigned int cycle, const std::string method, TableHandler results_table);

    void set_all_subdomain_objects (const std::vector<std::shared_ptr<Step6<dim>>> &objects)
    { subdomain_objects = objects; }

    // std::vector<Vector<double>> solution_vector;
    std::vector<std::unique_ptr<Functions::FEFieldFunction<dim>>> solutionfunction_vector;



private:
    void setup_system();
    void assemble_system(const std::string method);
    Functions::FEFieldFunction<dim> & get_fe_function(const unsigned int boundary_id);
    void solve(TableHandler results_table);
    void refine_grid(TableHandler results_table);

    void output_results(const unsigned int cycle) const;

    const unsigned int s;
    std::vector<std::shared_ptr<Step6<dim>>> subdomain_objects;

    Triangulation<dim> triangulation;

    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;
    SparseMatrix<double> system_matrix;
    SparsityPattern      sparsity_pattern;

    Vector<double> solution;
    Vector<double> system_rhs;


};

// @sect3{The (<code>MyOverlappingBoundaryValues) class template}

// We need to provide interpolate_boundary_values() in Step6::assemble_system a way to
// compute the value to impose as a boundary condition for additive Schwarz. Additive
// Schwarz does not just take one subdomain's solution as the boundary condition on an
// egde as in multiplicative Schwarz; it uses a linear combination of multiple solutions.
// We cannot provide this as a function because the solutions from each subdomain are
// defined on separate triangulations with their own dof_handlers Instead, we can
// provide a way for the boundary condition to be directly computed from these separate
// solutions when given any point in the overlapping region of their domain. In
// assemble_system(), interpolate_boundary_values() provides the Point<dim> to the
// MyOverlappingBoundaryValues object to evaluate the value of the boundary condition
// for that particular point.

// My implementation of Additive Schwarz is broken up in the following way:
//      - the MyOverlappingBoundaryValues class immediately below
//      - the get_overlapping_solution_functions function is directly below the
//          get_fe_function function used to get the solutions to impose as
//          boundary conditions.
//      - the usage of get_overlapping_solution_functions() is in assemble_system()
//          to impose boundary conditions
//      - the initialization of MyOverlappingBoundaryValues objects is done in
//          assemble_system(), immediately before
//          using objects of this type to impose BC for additive Schwarz
// The debugging process is still on-going.


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

// Here we have a few additions. First is the initialization of the subdomain on which
// we are preparing to solve.

template <int dim>
Step6<dim>::Step6(const unsigned int s)
        : s(s),
        fe(2),
        dof_handler(triangulation)

{

// Then we manually create the subdomains using their defining corner points:

    const std::vector<Point<2>> corner_points = {
            Point<2>(-0.875, -0.125), Point<2>(0.125, 0.875),
            Point<2>(-0.125, -0.125), Point<2>(0.875, 0.875),
            Point<2>(-0.125,-0.875), Point<2>(0.875,0.125),
            Point<2>(-0.875,-0.875), Point<2>(0.125,0.125)};



    GridGenerator::hyper_rectangle(triangulation,
                                   corner_points[2 * s],
                                   corner_points[2 * s + 1]);

    triangulation.refine_global(2);



// Lastly, we set boundary_ids (those not explicitly set to a value are
// zero by default, a fact that we use later):

    //Set the boundary_ids of edges of subdomain0
    // if (subdomain == 0) {
    if (s == 0) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Right edge:

                //Set boundary_id to 2 along the portion of gamma2 that makes
                // up part of the right boundary edge of subdomain0
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) >= 0.125))
                    face->set_boundary_id(2);

                //Set boundary_id to 6 along gamma6, the remaining portion of
                // subdomain0's right edge
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) < 0.125))
                    face->set_boundary_id(6);

                //Bottom edge:

                //Set boundary_id to 4 along the portion of gamma4 that makes
                // up part of the bottom boundary edge of subdomain0
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) < -0.125))
                    face->set_boundary_id(4);

                //Set boundary_id to 8 along gamma8, the remaining portion of
                // subdomain0's bottom edge
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) >= -0.125))
                    face->set_boundary_id(8);

                //Remaining edges have boundary_ids of 0 by default.

            }

    //Set the boundary_ids of edges of subdomain1
    //} else if (subdomain == 1) {
    } else if (s == 1) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Left edge:

                //Set boundary_id to 1 along portion of gamma1 that makes
                // up part of the left boundary edge of subdomain1
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) >= 0.125))
                    face->set_boundary_id(1);

                //Set boundary_id to 5 along gamma5, the remaining portion of
                // subdomain1's left edge
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) < 0.125))
                    face->set_boundary_id(5);

                //Bottom edge:

                //Set boundary_id to 4 along portion of gamma4 that makes
                // up part of the bottom boundary edge of subdomain1
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) >= 0.125))
                    face->set_boundary_id(4);

                //Set boundary_id to 8 along gamma8, the remaining portion of
                // subdomain1's bottom edge
                if ((std::fabs(center(dim - 1) - (-0.125)) < 1e-12) &&
                    (center(0) < 0.125))
                    face->set_boundary_id(8);

                //Remaining edges have boundary_ids of 0 by default.

            }

    //Set the boundary_ids of edges of subdomain2
    //} else if (subdomain == 2) {
    } else if (s == 2) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Left edge:

                //Set boundary_id to 1 along portion of gamma1 that makes
                // up part of the left boundary edge of subdomain2
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) <= -0.125))
                    face->set_boundary_id(1);

                //Set boundary_id to 5 along gamma5, the remaining portion of
                // subdomain2's left edge
                if ((std::fabs(center(0) - (-0.125)) < 1e-12) &&
                    (center(dim - 1) > -0.125))
                    face->set_boundary_id(5);

                //Top edge:

                //Set boundary_id to 3 along portion of gamma3 that makes
                // up part of the top boundary edge of subdomain2
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) >= 0.125))
                    face->set_boundary_id(3);

                //Set boundary_id to 7 along gamma7, the remaining portion of
                // subdomain2's top edge
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) < 0.125))
                    face->set_boundary_id(7);

                //Remaining edges have boundary_ids of 0 by default.

            }

    //Set the boundary_ids of edges of subdomain3
    //} else if (subdomain == 3) {
    } else if (s == 3) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();

                //Right edge:

                //Set boundary_id to 2 along portion of gamma2 that makes
                // up part of the right boundary edge of subdomain3
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) <= -0.125))
                    face->set_boundary_id(2);

                //Set boundary_id to 6 along gamma6, the remaining portion of
                // subdomain3's right edge
                if ((std::fabs(center(0) - (0.125)) < 1e-12) &&
                    (center(dim - 1) > -0.125))
                    face->set_boundary_id(6);

                //Top edge:

                //Set boundary_id to 3 along portion of gamma3 that makes
                // up part of the top boundary edge of subdomain3
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) <= -0.125))
                    face->set_boundary_id(3);

                //Set boundary_id to 7 along gamma7, the remaining portion
                // of subdomain3's top edge
                if ((std::fabs(center(dim - 1) - (0.125)) < 1e-12) &&
                    (center(0) > -0.125))
                    face->set_boundary_id(7);

                //Remaining edges have boundary_ids of 0 by default.

            }

    } else
        Assert (false, ExcInternalError());


    // Setting the boundary_ids as above is clearly a lengthy, tedious, and bug-prone
    // process that makes the program, as it currently stands, combersome to use.
    // This process will eventually be replaced this with an automated method, but
    // I needed a highly controlled system running before playing with this feature.


    setup_system();

    // (<code> setup_system) makes (<code> solution) of the right size, (<code> n_dofs),
    // and distributes dofs. So after (<code> setup_system), we can make a
    // FEFieldFunction (using solution and dof_handler) equivalent to the zero function
    // (because solution is initialized with a value of zero by default) and make this
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

    VectorTools::interpolate_boundary_values(
            dof_handler,0,
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

// Retrieves the appropriate (<code> FEFieldFunction) function to be imposed as a
// boundary condition in the multiplicative Schwarz algorithm.
// This association between subdomains, which subdomain solution is imposed as a
// boundary condition to which edge, is also done explicitly here, but will be
// replaced with an automated system.


template<int dim>
Functions::FEFieldFunction<dim> &
        Step6<dim>::get_fe_function(unsigned int boundary_id)
{

    // Get other subdomain id
    types::subdomain_id relevant_subdomain;

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


        //std::cout << "              The BC function for subdomain " << s <<
        // " on gamma" << boundary_id << " is from subdomain " <<
        // relevant_subdomain << std::endl;

        //For Multiplicative Schwarz, we impose the most recently computed solution
        // from neighboring subdomains as the BC of the current subdomain, so we
        // retrieve the last entry of the appropriate solutionfunction_vector:

        return *subdomain_objects[relevant_subdomain]->solutionfunction_vector.back();




}

// @sect4{MyOverlappingBoundaryValues<dim>::get_overlapping_solution_functions}

// Retrieves the appropriate (<code> FEFieldFunction) functions to be imposed as a
// boundary conditions in the additive Schwarz algorithm.
// This association between subdomains, which subdomain solution is imposed as a
// boundary condition to which edge, is also done explicitly here, but will be
// replaced with an automated system.


// @sect4{Step6::assemble_system}

// Next, we actually impose the appropriate solutions as boundary conditions on their
// appropriate edges, which is done with VectorTools::interpolate_boundary_values.
// We specify the function that will be applied as a boundary condition on the nonzero
// boundary_ids using get_fe_function for multiplicative Schwarz and
// get_overlapping_solution_functions for additive Schwarz. but on the boundaries
// with boudary_id of zero by default (the edges on the exterior of the entire domain)
// we impose the homogenous Direchlet boundary condition by using
// Functions::ZeroFunction<dim>() as the appropriate boundary function.


template <int dim>
void Step6<dim>::assemble_system(const std::string method)
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
                cell_matrix, cell_rhs, local_dof_indices,
                system_matrix, system_rhs);
    }


    std::map<types::global_dof_index, double> boundary_values;

    //Impose boundary condition on edges with boundary_id of 0
    VectorTools::interpolate_boundary_values(
            dof_handler,
            0,
            Functions::ZeroFunction<dim>(),
            boundary_values);


    //Impose boundary conditions for multiplicative Schwarz

    if (method == "Multiplicative") {


        //Impose boundary conditions on edges of subdomain0 with nonzero
        // boundary_ids
        if (s == 0) {

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    2,
                    get_fe_function(2),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    4,
                    get_fe_function(4),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    6,
                    get_fe_function(6),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    8,
                    get_fe_function(8),
                    boundary_values);


        //Impose boundary conditions on edges of subdomain1 with nonzero
        // boundary_ids
        } else if (s == 1) {

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    1,
                    get_fe_function(1),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    4,
                    get_fe_function(4),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    5,
                    get_fe_function(5),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    8,
                    get_fe_function(8),
                    boundary_values);

        //Impose boundary conditions on edges of subdomain2 with nonzero
        // boundary_ids
        } else if (s == 2) {
            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    1,
                    get_fe_function(1),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    3,
                    get_fe_function(3),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    5,
                    get_fe_function(5),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    7,
                    get_fe_function(7),
                    boundary_values);

            //Impose boundary conditions on edges of subdomain3 with nonzero
            // boundary_ids
        } else if (s == 3) {
            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    2,
                    get_fe_function(2),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    3,
                    get_fe_function(3),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    6,
                    get_fe_function(6),
                    boundary_values);

            VectorTools::interpolate_boundary_values(
                    dof_handler,
                    7,
                    get_fe_function(7),
                    boundary_values);

        } else Assert (false, ExcInternalError());
        

    } else { //Neither Multiplicative Schwarz nor Additive Schwarz were
        // chosen as the solving method
        std::cout <<
        "Error: 'Multiplicative' or 'Additive' must be chosen as the solving method"
        << std::endl;
        Assert (false, ExcInternalError());

    }

    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);


}


// @sect4{Step6::solve}

// The same solver as before is used because this is not the focus of this
// modified program.

template <int dim>
void Step6<dim>::solve(TableHandler results_table)
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

// Here we store the solution as function that can be retrieved elsewhere.
// Namely, this will be used by get_fe_function and get_overlapping_solution_functions.
    std::unique_ptr<Functions::FEFieldFunction<dim>> solutionfunction_pointer =
            std::make_unique<Functions::FEFieldFunction<dim>>(dof_handler, solution);

    solutionfunction_vector.emplace_back (std::move(solutionfunction_pointer));

    std::cout << "  max solution value=" << solution.linfty_norm()
              << std::endl;
    results_table.add_value("MaxSolValue", solution.linfty_norm()); //////////////////////////////////////////table

}



// @sect4{Step6::refine_grid}

template <int dim>
void Step6<dim>::refine_grid(TableHandler results_table)
{
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                       solution,
                                       estimated_error_per_cell);

    std::cout << " Max error present:" << estimated_error_per_cell.linfty_norm() <<
    std::endl;
    results_table.add_value("MaxError", estimated_error_per_cell.linfty_norm()); ///////////////////////////////////table

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);

    triangulation.execute_coarsening_and_refinement();


}



// @sect4{Step6::output_results}

template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{

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

// The (<code> run) function remains nearly identical; some of the functions it calls
// merely needed extra arguments.

template <int dim>
void Step6<dim>::run(const unsigned int cycle, const std::string method, TableHandler results_table) {

    std::cout << "Cycle:  " << cycle << std::endl;
    std::cout << "Subdomain:  " << s << std::endl;

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    refine_grid(results_table);

    setup_system();

    std::cout << " After calling refine_grid():" << std::endl;

    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    results_table.add_value("ActiveCells", triangulation.n_active_cells()); ////////////////////////////////// table

    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;
    results_table.add_value("DoF", dof_handler.n_dofs());  /////////////////////////////////////////////////// table


    //assemble_system(s, method);
    assemble_system(method);


        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() <<
        std::endl;

    solve(results_table);

    output_results(cycle);

}


int main()
{
    try
    {
        // We now construct each subdomain problem and store them all in a vector
        // to solve iteratively.
        std::vector<std::shared_ptr<Step6<2>>> subdomain_problems;

        subdomain_problems.push_back (std::make_shared<Step6<2>> (0));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (1));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (2));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (3));


        // Tell each of the objects representing one subdomain each about the objects
        // representing all of the other subdomains:
        for (unsigned int s=0; s<subdomain_problems.size(); ++s) {
            subdomain_problems[s] -> set_all_subdomain_objects(subdomain_problems);
        }

        // Choose whether we use multiplicative or additive Schwarz to solve
        std::string method;
        std::cout <<
        "Which Schwarz method would you like to use? (Multiplicative or Additive)" <<
        std::endl;

        std::cin >> method;

        TableHandler results_table; //////////////////////////////////////////////////////////////////////////////////// table
        Timer timer;

        // Now we can actually solve each subdomain problem
        for (unsigned int cycle=1; cycle<10; ++cycle)
            for (unsigned int s=0; s<subdomain_problems.size(); ++s) {
                subdomain_problems[s] -> run(cycle, method, results_table);

                std::cout << "After solving on subdomain " << s <<
                " during cycle " << cycle << ":\n" <<
                "        Elapsed CPU time: " << timer.cpu_time() <<
                " seconds.\n";

                std::cout << "        Elapsed wall time: " << timer.wall_time() <<
                " seconds.\n";


                results_table.add_value("Cycle", cycle);                      ////////////////////////////////////// table
                results_table.add_value("Subdomain", s);                      ////////////////////////////////////// table
                results_table.add_value("CPUtimeTotal", timer.cpu_time());    ////////////////////////////////////// table
                results_table.add_value("WalltimeTotal", timer.wall_time());  ////////////////////////////////////// table

            }

        timer.stop();
        timer.reset();

        std::ofstream out_file("number_table.tex");
        results_table.write_tex(out_file);
        out_file.close();

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




