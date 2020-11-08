


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


using namespace dealii;

template <int dim>
class Step6
{
public:
    Step6(const unsigned subdomain);
    void run();

    void set_all_subdomain_objects (std::vector<shared_ptr<Step6<dim>>> &objects)
    { subdomain_objects = objects; }



private:
    void setup_system(unsigned int subdomain);
    void assemble_system(unsigned int subdomain, const unsigned int cycle);
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle) const;

    const unsigned int my_subdomain;

    std::vector<shared_ptr....> subdomain_objects; // subdomain_ids,


    Triangulation<dim> triangulation;


    FE_Q<dim>       fe;
    DoFHandler<dim> dof_handler;

    AffineConstraints<double> constraints;
    SparseMatrix<double> system_matrix;
    SparsityPattern      sparsity_pattern;

    Vector<double> solution;
    Vector<double> system_rhs;

    //Want to store solutions for each subdomain to be retrieved later...
    std::vector<std::unique_ptr<Functions::FEFieldFunction<dim>>> solutionfunction_vector;

};


template <int dim>
double coefficient(const Point<dim> &p)
{
    if (p.square() < 0.5 * 0.5)
        return 20;
    else
        return 1;
}


template <int dim>
Step6<dim>::Step6(const unsigned int subdomain)
        :  my_subdomain (subdomain) ////////////////////////
        , fe(2)
        , dof_handler(triangulation)

{
    //Create separate triangulations for each subdomain:
    const std::vector<Point<2>> corner_points = {Point<2>(-1, -1), Point<2>(0.25, 1),
                                                 Point<2>(-0.25, -1), Point<2>(1, 1)};

    GridGenerator::hyper_rectangle(triangulation, corner_points[2 * subdomain],
                                   corner_points[2 * subdomain + 1]);

    triangulation.refine_global(1);

    //set the boundary_id to 2 along gamma2:
    if (subdomain == 0) {
        for (const auto &cell : triangulation.cell_iterators())
            for (const auto &face : cell->face_iterators()) {
                const auto center = face->center();
                if ((std::fabs(center(0) - (0.25)) < 1e-12))
                    face->set_boundary_id(2);
            }

    } else { //we are in subdomain 1 and set the boundary_id to 1 along gamma1:
        for (const auto &cell : triangulation.cell_iterators())
        for (const auto &face : cell->face_iterators()) {
            const auto center = face->center();
            if ((std::fabs(center(0) - (-0.25)) < 1e-12))
                face->set_boundary_id(1);
        }
    }


}




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

//Need function to get the appropriate solution fe_function.
// Specifically, with only two subdomains, we know that this appropriate fe_function is the one
// created from the previously computed solution, and therefore, is the last entry of our vector of solutions:
//

template<int dim>
Functions::FEFieldFunction<dim>::get_fe_function(unsigned int boundary_id, unsigned int cycle)
{
    //Since we know that we only have two subdomains, we know that the last entry of the solution vector is that
    //of the other subdomain which we need to impose as a BC of the current subdomain:
    fe_function = solutionfunction_vector[solutionfunction_vector.size()-1];

    //More generally, we will need to retrieve the
    // {[(# of subdomains)*(cycle-1)]+(subdomain# whose solution will be the BC-cycle)}th entry:

    //fe_function = solutionfunction_vector[subdomain_problems.size()*(cycle-1)+(relevant subdomain number)]
    //relevant subdomain number can be retrieved using the boundary_id#

    /*
    int relevant_subdomain;

    if (boundary_id == 1){
        relevant_subdomain = 1;
    } else if (boundary_id == 2){
        relevant_subdomain = 0;
    } else //boundary_id = 0
        relevant_subdomain = -1;

    if (relevant_subdomain == -1) { //equivalently checking if boundary_id = 0
        fe_function = ZeroFunction //How do I make this of type FEFieldFunction?
    } else {
        fe_function = solutionfunction_vector[subdomain_problems.size()*(cycle - 1)+(relevant_subdomain)]

    }
    */

    return fe_function;
}

template <int dim>
void Step6<dim>::assemble_system(unsigned int subdomain, unsigned int cycle)
{
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


    //Functions::FEFieldFunction<dim> fe_function(dof_handler, solution);


    std::map<types::global_dof_index, double> boundary_values;


    /*if (cycle == 0 && subdomain == 0){
        //When cycle=0 and subdomain=0, the only boundary_ids that exist in our system are 0 and 2
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 Functions::ZeroFunction<dim>(),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 2,
                                                 Functions::ZeroFunction<dim>(),
                                                 boundary_values);


    } else if (subdomain == 1) {

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 Functions::ZeroFunction<dim>(),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 1,
                                                 fe_function, //need to use get_fe_function() here
                                                 boundary_values);


    } else { //subdomain == 0, cycle > 1
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 0,
                                                 Functions::ZeroFunction<dim>(),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 2,
                                                 fe_function1, //need to use get_fe_function() here
                                                 boundary_values);

    }*/
    //Do not need to check for subdomains anymore. Regardless of which subdomain we are setting up to solve, the
    //boundary condition will come from the solution last computed.
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             //fe_function,
                                             get_fe_function(1, cycle),
                                             boundary_values);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             2,
                                             get_fe_function(2, cycle),
                                             boundary_values);
            //When subdomain 0 is created, there is no edge with boundary_id=2.
            // Is this a problem or does interpolate_boundary_id() check for boundary_ids, apply the given function
            // to the edge with this id, and do nothing when the given boundary_id is not found?

            //Also, after subdomain1 is created, we have now assigned boundary_id=2 to gamma2.
            // When solving on subdomain0, the edge with boundary_id=2 is not a boundary edge of subdomain0,
            // so even if we have:
            //                  VectorTools::interpolate_boundary_values(dof_handler, 2, ...)
            // I would hope that interpolate_boundary_values() does not try to apply boundary conditions on gamma2.
            //Is this the case?

            //Similarly, when solving on subdomain1, gamma1 is not a boundary edge of the current subdomain and I would
            // hope that no conditions are enforced over this edge.

            //If interpolate_boundary_values enforces conditions on ANY edge with the provided boundary_id, I will need
            // to know which subdomain I am solving on so that I only impose conditions on the appropriate edges.


    MatrixTools::apply_boundary_values(boundary_values,
                                       system_matrix,
                                       solution,
                                       system_rhs);


}



template <int dim>
void Step6<dim>::solve()
{
    SolverControl solver_control(1000, 1e-12);
    SolverCG<Vector<double>> solver(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    solver.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);

//Store solution as function that can be retrieved elsewhere:
    //std::vector<std::unique_ptr<Functions::FEFieldFunction<dim>>> solutionfunction_vector; //declared elsewhere
    std::unique_ptr<Functions::FEFieldFunction<dim>> pointer =
            std::make_unique<Functions::FEFieldFunction<dim>>(dof_handler, solution);
    solutionfunction_vector.emplace_back(pointer);

    //So we are storing only the solution as a function, not storing the associated dof_handler...
    //Since each subdomain has a separate triangulation, will we actually need to store the dof_handler along with the
    // solution function in case we need to translate from one dof_handler to another?

}



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



template <int dim>
void Step6<dim>::output_results(const unsigned int cycle) const
{
    {

        GridOut grid_out;
        std::ofstream output("grid-" + std::to_string(cycle) + ".gnuplot");
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
        std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
        data_out.write_vtu(output);

    }
}


template<int dim>
void set_all_subdomain_objects(std::vector<std::shared_ptr<Step6<2>>> subdomain_problems)
{


}




template <int dim>
void Step6<dim>::run() {

    for (unsigned int cycle = 1; cycle < 8; ++cycle) { //in cycle=0, we did all of the set up (in the constructor)

    /*
        if (cycle == 0) {
            GridGenerator::hyper_rectangle(triangulation, corner_points[2 * subdomain],
                                           corner_points[2 * subdomain + 1]);
            triangulation.refine_global(1);

            //set the boundary_id to 2 along gamma2:
            for (const auto &cell : triangulation.cell_iterators())
                for (const auto &face : cell->face_iterators()) {
                    const auto center = face->center();
                    if ((std::fabs(center(0) - (0.25)) < 1e-12))
                        face->set_boundary_id(2);
                }
*/ //moved this to Step6 constructor

            refine_grid();
            std::cout << "   Number of active cells:       "
                      << triangulation.n_active_cells() << std::endl;
            setup_system();
            std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
                      << std::endl;
            assemble_system(cycle);
            solve();
            output_results(cycle);

    }
}



int main()
{
    try
    {
        Step6<2> laplace_problem_2d;
        //laplace_problem_2d.run();

        std::vector<std::shared_ptr<Step6<2>>> subdomain_problems;
        subdomain_problems.push_back (std::make_shared<...> (0));
        subdomain_problems.push_back (std::make_shared<...> (1));

        for (unsigned int s=0; s<subdomain_problems.size(); ++s)
            subdomain_problems[s] -> set_all_subdomain_objects(subdomain_problems);


        //for(iteration...)
        for (unsigned int cycle = 1; cycle < 8; ++cycle)
            for (unsigned int s=0; s<subdomain_problems.size(); ++s) {
                ...update subdomain s in iteration i;
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




