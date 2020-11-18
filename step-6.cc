


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
    Step6(const unsigned int subdomain);
    void run(const unsigned int cycle, const unsigned int s);

    void set_all_subdomain_objects (const std::vector<std::shared_ptr<Step6<dim>>> &objects)
    { subdomain_objects = objects; }

    std::vector<std::unique_ptr<Functions::FEFieldFunction<dim>>> solutionfunction_vector;



private:
    void setup_system();
    void assemble_system(const unsigned int cycle, const unsigned int s);
    Functions::FEFieldFunction<dim> & get_fe_function(const unsigned int boundary_id, const unsigned int cycle);
    void solve();
    void refine_grid();
    void output_results(const unsigned int cycle, const unsigned int s) const;

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

        : subdomain (subdomain),
        fe(2),
        dof_handler(triangulation)

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

    setup_system();


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

//Need function to get the appropriate solution fe_function:
template<int dim>
Functions::FEFieldFunction<dim> &
        Step6<dim>::get_fe_function(unsigned int boundary_id, unsigned int cycle)
{

    // get other subdomain id
    types::subdomain_id relevant_subdomain;

    //only handle nonzero boundary_ids for now:
    if (boundary_id == 1){
        relevant_subdomain = 0;
    } else if (boundary_id == 2){
        relevant_subdomain = 1;
    } else
        relevant_subdomain = -1;

    //For Multiplicative Schwarz, we impose the most recently computed solution from neighboring subdomains as the
    // BC of the current subdomain, so we retrieve the last entry of the appropriate solutionfunction_vector:
    return *subdomain_objects[relevant_subdomain] -> solutionfunction_vector.back();

    //Later, for Additive Schwarz, we will sometimes need to access the second to last element of a
    // solutionfunction_vector.

}

template <int dim>
void Step6<dim>::assemble_system(unsigned int cycle, unsigned int s)
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

    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(),
                                             boundary_values);

    if (cycle == 1){
         if (s==0) { //solutionfunction_vector of subdomain1 is empty!
             // We can work around the need for this if-else block later by having the first entry of
             // each solutionfunction_vector be the ZeroFunction of type FEFieldFunction...
             VectorTools::interpolate_boundary_values(dof_handler,
                                                      2,
                                                      Functions::ZeroFunction<dim>(),
                                                      boundary_values);

         } else { //s=1, here we use the most recent solution from subdomain0
             VectorTools::interpolate_boundary_values(dof_handler,
                                                      1,
                                                      get_fe_function(1, cycle),
                                                      boundary_values);
         }

    } else { //now all solutionfunction_vectors have at least has one entry that we can retrieve with get_fe_function()
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 1,
                                                 get_fe_function(1, cycle),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 2,
                                                 get_fe_function(2, cycle),
                                                 boundary_values);
    }


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
    std::unique_ptr<Functions::FEFieldFunction<dim>> pointer =
            std::make_unique<Functions::FEFieldFunction<dim>>(dof_handler, solution);
    solutionfunction_vector.emplace_back (std::move(pointer));

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
void Step6<dim>::output_results(const unsigned int cycle, const unsigned int s) const
{
    {

        GridOut grid_out;

        //std::ofstream output("grid-" + std::to_string(cycle) + ".gnuplot");
        std::ofstream output("grid-" + std::to_string(cycle) + "-" + std::to_string(s)  +".gnuplot");

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

        //std::ofstream output("solution-" + std::to_string(cycle) + ".vtu");
        std::ofstream output("solution-" + std::to_string(cycle) + "-" + std::to_string(s) + ".vtu");

        data_out.write_vtu(output);

    }
}



template <int dim>
void Step6<dim>::run(const unsigned int cycle, const unsigned int s) {

    //refine_grid(); //incorporate this later
    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells() << std::endl;
    std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl;

    assemble_system(cycle, s);

        //For debugging purposes only:
        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    solve();

    output_results(cycle, s);

        //For debugging purposes only:
        std::cout << "Cycle:  " << cycle << std::endl;
        std::cout << "Subdomain:  " << s << std::endl;


}



int main()
{
    try
    {

        std::vector<std::shared_ptr<Step6<2>>> subdomain_problems;
        subdomain_problems.push_back (std::make_shared<Step6<2>> (0));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (1));

        // Tell each of the objects representing one subdomain each about
        // the objects representing all of the other subdomains
        for (unsigned int s=0; s<subdomain_problems.size(); ++s)
            subdomain_problems[s] -> set_all_subdomain_objects(subdomain_problems);


        for (unsigned int cycle = 1; cycle < 8; ++cycle)
            for (unsigned int s=0; s<subdomain_problems.size(); ++s) {
                subdomain_problems[s] -> run(cycle, s);
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




