


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

    //const std::vector<Point<2>> corner_points;




private:
    void setup_system();
    void assemble_system(const unsigned int s);
    Functions::FEFieldFunction<dim> & get_fe_function(const unsigned int boundary_id, const unsigned int s);
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

    Vector<double> solution; //initialized with default value of 0
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

//Create separate triangulations for each subdomain: --------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------
// 2 subdomains case:
    //const std::vector<Point<2>> corner_points = {Point<2>(-1, -1), Point<2>(0.25, 1),
    //                                             Point<2>(-0.25, -1), Point<2>(1, 1)};

// ----------------------------------------------------------------------------------------------------------------

//4 subdomains case:
    const std::vector<Point<2>> corner_points = {Point<2>(0, 0.75), Point<2>(1, 1.75),
                                                 Point<2>(0.75, 0.75), Point<2>(1.75, 1.75),
                                                 Point<2>(0.75,0), Point<2>(1.75,1),
                                                 Point<2>(0,0), Point<2>(1,1)};


    GridGenerator::hyper_rectangle(triangulation, corner_points[2 * subdomain],
                                   corner_points[2 * subdomain + 1]);

    triangulation.refine_global(2);



// Set boundary_ids: ----------------------------------------------------------------------------------------------


// ----------------------------------------------------------------------------------------------------------------
//2 subdomain case
    //set the boundary_id to 2 along gamma2:
   /* if (subdomain == 0) {
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
*/
// ----------------------------------------------------------------------------------------------------------------

//4 subdomain case:

    //set the boundary_ids of edges of subdomain0
     if (subdomain == 0) {
         for (const auto &cell : triangulation.cell_iterators())
             for (const auto &face : cell->face_iterators()) {
                 const auto center = face->center();

             //Right edge:

                 //set boundary_id to 2 along the portion of gamma2 that makes up part of the right
                 // boundary edge of subdomain0

                 //if ((std::fabs(center(0) - (0.25)) < 1e-12) &&
                 // The center of a cell will not be within 1e-12 of its bounding value! The cells are bigger than this!
                 // Because of how our subdomains are constructed and because we initially refine their triangulation
                 // twice, each cell is a 0.25x0.25 square. Therefore, the center of a cell can only be within 0.25/2
                 // or 0.125 from any bounding x or y value.
                 // To check that we are on a cell adjacent to an edge gamma, it is therefore sufficient to check that
                 // the center is strictly within 0.25 of this edge's defining x or y value.
                 // This value depends entirely on how much refinement is done initially.

                 if ((std::fabs(center(0) - (1)) < 1e-12) && // I don't understand why '< 1e-12 works', thoughts above...
                         (center(dim - 1) > 1))
                     face->set_boundary_id(2);

                 //set boundary_id to 6 along gamma6, the remaining portion of subdomain0's right edge
                 if ((std::fabs(center(0) - (1)) < 1e-12) &&
                     (center(dim - 1) <= 1))
                     face->set_boundary_id(6);

             //Bottom edge:

                 //set boundary_id to 4 along the portion of gamma4 that makes up part of the bottom
                 // boundary edge of subdomain0
                 if ((std::fabs(center(dim - 1) - (0.75)) < 1e-12) &&
                     (center(0) < 0.75))
                     face->set_boundary_id(4);

                 //set boundary_id to 8 along gamma8, the remaining portion of subdomain0's bottom edge
                 if ((std::fabs(center(dim - 1) - (0.75)) < 1e-12) &&
                     (center(0) >= 0.75))
                     face->set_boundary_id(8);

             //Remaining edges have boundary_ids of 0 by default.

             }

         std::cout << "              Properly set boundary_ids of subdomain 0 " <<  std::endl;

     //set the boundary_ids of edges of subdomain1
     } else if (subdomain == 1) {
         for (const auto &cell : triangulation.cell_iterators())
             for (const auto &face : cell->face_iterators()) {
                 const auto center = face->center();

             //Left edge:

                 //set boundary_id to 1 along portion of gamma1 that makes up part of the left
                 // boundary edge of subdomain1
                 if ((std::fabs(center(0) - (0.75)) < 1e-12) &&
                     (center(dim - 1) > 1))
                     face->set_boundary_id(1);

                 //set boundary_id to 5 along gamma5, the remaining portion of subdomain1's left edge
                 if ((std::fabs(center(0) - (0.75)) < 1e-12) &&
                     (center(dim - 1) <= 1))
                     face->set_boundary_id(5);

             //Bottom edge:

                 //set boundary_id to 4 along portion of gamma4 that makes up part of the bottom
                 // boundary edge of subdomain1
                 if ((std::fabs(center(dim - 1) - (0.75)) < 1e-12) &&
                     (center(0) > 1))
                     face->set_boundary_id(4);

                 //set boundary_id to 8 along gamma8, the remaining portion of subdomain1's bottom edge
                 if ((std::fabs(center(dim - 1) - (0.75)) < 1e-12) &&
                     (center(0) <= 1))
                     face->set_boundary_id(8);

             //Remaining edges have boundary_ids of 0 by default.

             }

     //set the boundary_ids of edges of subdomain2
     } else if (subdomain == 2) {
         for (const auto &cell : triangulation.cell_iterators())
             for (const auto &face : cell->face_iterators()) {
                 const auto center = face->center();

             //Left edge:

                 //set boundary_id to 1 along portion of gamma1 that makes up part of the left
                 // boundary edge of subdomain2
                 if ((std::fabs(center(0) - (0.75)) < 1e-12) &&
                     (center(dim - 1) < 0.75))
                     face->set_boundary_id(1);

                 //set boundary_id to 5 along gamma5, the remaining portion of subdomain2's left edge
                 if ((std::fabs(center(0) - (0.75)) < 1e-12) &&
                     (center(dim - 1) >= 0.75))
                     face->set_boundary_id(5);

             //Top edge:

                 //set boundary_id to 3 along portion of gamma3 that makes up part of the top
                 // boundary edge of subdomain2
                 if ((std::fabs(center(dim - 1) - (1)) < 1e-12) &&
                     (center(0) > 1))
                     face->set_boundary_id(3);

                 //set boundary_id to 7 along gamma7, the remaining portion of subdomain2's top edge
                 if ((std::fabs(center(dim - 1) - (1)) < 1e-12) &&
                     (center(0) <= 1))
                     face->set_boundary_id(7);

             //Remaining edges have boundary_ids of 0 by default.

             }

     //set the boundary_ids of edges of subdomain3
     } else if (subdomain == 3) {
         for (const auto &cell : triangulation.cell_iterators())
             for (const auto &face : cell->face_iterators()) {
                 const auto center = face->center();

             //Right edge:

                 //set boundary_id to 2 along portion of gamma2 that makes up part of the right
                 // boundary edge of subdomain3
                 if ((std::fabs(center(0) - (1)) < 1e-12) &&
                     (center(dim - 1) < 0.75))
                     face->set_boundary_id(2);

                 //set boundary_id to 6 along gamma6, the remaining portion of subdomain3's right edge
                 if ((std::fabs(center(0) - (1)) < 1e-12) &&
                     (center(dim - 1) >= 0.75))
                     face->set_boundary_id(6);

             //Top edge:

                 //set boundary_id to 3 along portion of gamma3 that makes up part of the top
                 // boundary edge of subdomain3
                 if ((std::fabs(center(dim - 1) - (1)) < 1e-12) &&
                     (center(0) < 0.75))
                     face->set_boundary_id(3);

                 //set boundary_id to 7 along gamma7, the remaining portion of subdomain3's top edge
                 if ((std::fabs(center(dim - 1) - (1)) < 1e-12) &&
                     (center(0) >= 0.75))
                     face->set_boundary_id(7);

             //Remaining edges have boundary_ids of 0 by default.

         }

     } else
        Assert (false, ExcInternalError()); // always aborts the program


    setup_system(); //makes solution of the right size (n_dofs) and distributes dofs
                    // so after setup_system(), we can make a FEFieldFunction (using solution and dof_handler)
                    // equivalent to the zero function (because solution is iniitalized with a value of zero by default)
                    // and make this FEFieldfunction the first entry of each solutionfunction_vector:

//Make the first element in each solutionfunction_vector the zero_function:
    std::unique_ptr<Functions::FEFieldFunction<dim>> zero_function =
            std::make_unique<Functions::FEFieldFunction<dim>>(dof_handler, solution);
    solutionfunction_vector.emplace_back (std::move(zero_function));

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
        Step6<dim>::get_fe_function(unsigned int boundary_id, unsigned int s)
{

    // get other subdomain id
    types::subdomain_id relevant_subdomain;

// ----------------------------------------------------------------------------------------------------------------
// 2 subdomain case:
/*
    if (boundary_id == 1){
        relevant_subdomain = 0;
    } else if (boundary_id == 2){
        relevant_subdomain = 1;
    } else
        Assert (false, ExcInternalError()); // always aborts the program
*/

// ----------------------------------------------------------------------------------------------------------------


//4 subdomain case:

    if (s == 0){
        if (boundary_id == 2)
            relevant_subdomain = 1;

        //else if (boundary_id == 6 || boundary_id == 8)
        //    relevant_subdomain = 2;
        else if (boundary_id == 6)
            relevant_subdomain = 2;

        else if (boundary_id == 8)
            relevant_subdomain = 2;

        else if (boundary_id == 4)
            relevant_subdomain = 3;

        else //boundary_id == 0
            Assert (false, ExcInternalError()); //make sure to only use get_fe_function on nonzero boundary_ids!


    } else if (s == 1) {
        if (boundary_id == 1)
            relevant_subdomain = 0;

        //else if (boundary_id == 5 || boundary_id == 8)
        //    relevant_subdomain = 3;

        else if (boundary_id == 5)
            relevant_subdomain = 3;

        else if (boundary_id == 8)
            relevant_subdomain = 3;

        else if (boundary_id == 4)
            relevant_subdomain = 2;

        else //boundary_id == 0
            Assert (false, ExcInternalError());


    } else if (s == 2) {

        if (boundary_id == 3)
            relevant_subdomain = 1;

        //else if (boundary_id == 5 || boundary_id == 7)
        //    relevant_subdomain = 0;
        else if (boundary_id == 5)
            relevant_subdomain = 0;

        else if (boundary_id == 7)
            relevant_subdomain = 0;

        else if (boundary_id == 1)
            relevant_subdomain = 3;

        else //boundary_id == 0
            Assert (false, ExcInternalError());


    } else if (s == 3) {

        if (boundary_id == 3)
            relevant_subdomain = 0;

        //else if (boundary_id == 6 || boundary_id == 7)
        //    relevant_subdomain = 1;

        else if (boundary_id == 6)
            relevant_subdomain = 1;

        else if (boundary_id == 7)
            relevant_subdomain = 1;

        else if (boundary_id == 2)
            relevant_subdomain = 2;

        else //boundary_id == 0
            Assert (false, ExcInternalError());


    } else
        Assert (false, ExcInternalError()); // always aborts the program

    //For debugging:
    std::cout << "              The BC function for subdomain " << s << " on gamma" << boundary_id <<
                 " is from subdomain " << relevant_subdomain <<  std::endl;

    //For Multiplicative Schwarz, we impose the most recently computed solution from neighboring subdomains as the
    // BC of the current subdomain, so we retrieve the last entry of the appropriate solutionfunction_vector:
    return *subdomain_objects[relevant_subdomain] -> solutionfunction_vector.back();

    //Later, for Additive Schwarz, we will sometimes need to access the second to last element of a
    // solutionfunction_vector.

}




template <int dim>
void Step6<dim>::assemble_system(unsigned int s)
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

// ----------------------------------------------------------------------------------------------------------------
//2 subdomain case:
/*

     if (s==0) {
         VectorTools::interpolate_boundary_values(dof_handler,
                                                  2,
                                                  get_fe_function(1, s),
                                                  boundary_values);

     } else { //s=1
         VectorTools::interpolate_boundary_values(dof_handler,
                                                  1,
                                                  get_fe_function(1, s),
                                                  boundary_values);
     }

*/

// ----------------------------------------------------------------------------------------------------------------

//4 subdomain case:

    //Impose boundary conditions on edges of subdomain0 with nonzero boundary_ids
    if (s == 0){

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 2,
                                                 get_fe_function(2, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 4,
                                                 get_fe_function(4, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 6,
                                                 get_fe_function(6, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 8,
                                                 get_fe_function(8, s),
                                                 boundary_values);


    //Impose boundary conditions on edges of subdomain1 with nonzero boundary_ids
    } else if (s == 1){

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 1,
                                                 get_fe_function(1, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 4,
                                                 get_fe_function(4, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 5,
                                                 get_fe_function(5, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 8,
                                                 get_fe_function(8, s),
                                                 boundary_values);

    //Impose boundary conditions on edges of subdomain2 with nonzero boundary_ids
    } else if (s == 2){
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 1,
                                                 get_fe_function(1, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 3,
                                                 get_fe_function(3, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 5,
                                                 get_fe_function(5, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 7,
                                                 get_fe_function(7, s),
                                                 boundary_values);

    } else if (s == 3){
        VectorTools::interpolate_boundary_values(dof_handler,
                                                 2,
                                                 get_fe_function(2, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 3,
                                                 get_fe_function(3, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 6,
                                                 get_fe_function(6, s),
                                                 boundary_values);

        VectorTools::interpolate_boundary_values(dof_handler,
                                                 7,
                                                 get_fe_function(7, s),
                                                 boundary_values);

    } else
    Assert (false, ExcInternalError()); // always aborts the program

    //Want to modify interpolate_boundary_values() slightly so that it checks if the given boundary_id
    // is present on a boundary of the current subdomain:
    //      if so: apply the given boundary_function
    //      if not: do nothing
    // so that we can run the following, without need to check which subdomain we are on:


/*

    VectorTools::interpolate_boundary_values(dof_handler,
                                             1,
                                             get_fe_function(1, s),
                                             boundary_values);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             2,
                                             get_fe_function(2, s),
                                             boundary_values);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             3,
                                             get_fe_function(3, s),
                                             boundary_values);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             4,
                                             get_fe_function(4, s),
                                             boundary_values);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             5,
                                             get_fe_function(5, s),
                                             boundary_values);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             6,
                                             get_fe_function(6, s),
                                             boundary_values);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             7,
                                             get_fe_function(7, s),
                                             boundary_values);

    VectorTools::interpolate_boundary_values(dof_handler,
                                             8,
                                             get_fe_function(8, s),
                                             boundary_values);

    ... or better, with a loop going through all nonzero boundary_id values
*/


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

    std::cout << "  max solution value=" << solution.linfty_norm()
              << std::endl;

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

        std::ofstream output("grid-" + std::to_string(s*100 + cycle)  +".gnuplot");

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

        std::ofstream output("solution-" + std::to_string(s*100 + cycle) + ".vtu");

        data_out.write_vtu(output);

    }
}




template <int dim>
void Step6<dim>::run(const unsigned int cycle, const unsigned int s) {

    //For debugging purposes only:
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

    assemble_system(s);

        //For debugging purposes only:
        std::cout << "   Number of active cells:       "
                  << triangulation.n_active_cells() << std::endl;
        std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    solve();

    output_results(cycle, s);

}




int main()
{
    try
    {

        std::vector<std::shared_ptr<Step6<2>>> subdomain_problems;

        subdomain_problems.push_back (std::make_shared<Step6<2>> (0));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (1));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (2));
        subdomain_problems.push_back (std::make_shared<Step6<2>> (3));
        //Want to do this in a loop instead:

        /*
        for(unsigned int s=0; s<0.5*corner_points.size(); ++s){ //does not recognize corner_points
            subdomain_problems.push_back (std::make_shared<Step6<2>> (s));
        }
        */



        // Tell each of the objects representing one subdomain each about the objects representing all of the
        // other subdomains:
        for (unsigned int s=0; s<subdomain_problems.size(); ++s) {
            subdomain_problems[s] -> set_all_subdomain_objects(subdomain_problems);
        }


        for (unsigned int cycle=1; cycle<8; ++cycle) //only worked for 11 cycles with refinement
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




